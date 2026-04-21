import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

from ..geometry.projection import get_fov
from ..misc.cam_utils import camera_normalization
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

# Import FaMoS-specific utilities
from src.utils.camera import load_mpi_camera, rotate_image
from src.utils.utils import get_filename


@dataclass
class DatasetFaMoSCfg(DatasetCfgCommon):
    name: str
    data_root: Path
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    load_stereo: bool = False
    load_color: bool = True
    image_resize_factor: float = 1.0
    near: float = -1.0
    far: float = -1.0
    subset_fraction: float = 1.0  # fraction of the split to load (0.0, 1.0]


@dataclass
class DatasetFaMoSCfgWrapper:
    famos: DatasetFaMoSCfg


class DatasetFaMOS(Dataset):
    """Dataset loader for FaMoS data stored on disk.

    Reads calibration files (.tka), images, and split JSONs directly from
    the FaMoS directory structure, then converts each sample into the
    standard SPFSplat context/target dict format for the training pipeline.
    """

    cfg: DatasetFaMoSCfg
    stage: Stage
    view_sampler: ViewSampler
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetFaMoSCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Resolve data root and split
        data_root = Path(cfg.data_root)
        self.image_resize_factor = cfg.image_resize_factor
        self.load_stereo = cfg.load_stereo
        self.load_color = cfg.load_color

        # Determine split root and load split list
        split_root, data_list_fname = self._get_split_meta(data_root, stage)
        self.split_root = split_root
        self.split_list = self._load_json(data_list_fname)

        # Optionally sub-sample the split list
        if cfg.subset_fraction < 1.0:
            n = max(1, int(len(self.split_list) * cfg.subset_fraction))
            self.split_list = self.split_list[:n]
            print(f"[DatasetFaMOS] Using {n}/{len(self._load_json(data_list_fname))} "
                  f"samples ({cfg.subset_fraction*100:.1f}%) for stage={stage}")

        # Set up directory accessors
        self.dataset_root_dir = Path(data_list_fname).parent
        self.calibration_dir_base = split_root / "calibrations"
        self.image_dir = split_root / "downsampled_images_4"
        self.matting_dir = split_root / "matting"

        if not self.calibration_dir_base.exists():
            raise RuntimeError(f"Calibration directory not found: {self.calibration_dir_base}")

    # -------------------------------------------------------------------------
    # Split metadata
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_split_meta(famos_root: Path, stage: Stage):
        split_root = famos_root / "training_data"
        if stage == "test":
            split_root = famos_root / "test_data"

        if stage == "test":
            jsons = list(split_root.glob("*.json"))
            assert len(jsons) == 1, (
                f"Expected exactly one json file in {split_root}, found {len(jsons)}"
            )
            data_list_fname = str(jsons[0])
        else:
            json_kw = "val" if stage == "val" else "train"
            jsons = list(split_root.glob(f"*_{json_kw}.json"))
            assert len(jsons) == 1, (
                f"Expected exactly one json file matching '*_{json_kw}.json' in {split_root}, "
                f"found {len(jsons)}"
            )
            data_list_fname = str(jsons[0])

        return split_root, data_list_fname

    @staticmethod
    def _load_json(path: str):
        import json
        with open(path, "r") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # Path helpers (mirrors famos_dataloader.py)
    # -------------------------------------------------------------------------

    def _img_dir(self, subject: str, sequence: str, frame: str) -> str:
        return str(self.image_dir / subject / sequence / frame)

    def _img_fname(self, subject: str, sequence: str, frame: str, view: str) -> str:
        img_dir = self._img_dir(subject, sequence, frame)
        return os.path.join(img_dir, f"{sequence}.{frame}.{view}.png")

    def _matting_fname(self, subject: str, sequence: str, frame: str, view: str) -> str:
        matting_dir = self.matting_dir / subject / sequence / frame
        return os.path.join(matting_dir, f"{sequence}.{frame}.{view}.png")

    def _calibration_dir(self, subject: str, sequence: str) -> str:
        return str(self.calibration_dir_base / subject / sequence)

    # -------------------------------------------------------------------------
    # Image + camera reading (adapted from FaMosDataset._read_img_with_camera)
    # -------------------------------------------------------------------------

    def _read_img_with_camera(
        self,
        subject: str,
        sequence: str,
        frame: str,
        calib_fname: str,
        to_meters: bool = False,
    ):
        view_name = get_filename(calib_fname)
        image_fname = self._img_fname(subject, sequence, frame, view_name)
        matting_fname = self._matting_fname(subject, sequence, frame, view_name)

        if not os.path.exists(image_fname):
            return None

        try:
            image = imageio.imread(image_fname, pilmode="RGB")
        except Exception:
            print(f"Error loading image: {image_fname}")
            return None

        camera = load_mpi_camera(calib_fname, self.image_resize_factor, to_meters=to_meters)
        if camera is None:
            return None

        # Resize image if needed to match calibration
        if (image.shape[0] != camera["image_size"][0]) or (
            image.shape[1] != camera["image_size"][1]
        ):
            image = cv2.resize(
                image,
                (camera["image_size"][1], camera["image_size"][0]),
                interpolation=cv2.INTER_AREA,
            )
        
        # load matting
        if os.path.exists(matting_fname):
            matting = imageio.imread(matting_fname, pilmode="L")
            matting = cv2.resize(
                matting,
                (camera["image_size"][1], camera["image_size"][0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Rotate portrait images to landscape
        if camera["image_size"][0] > camera["image_size"][1]:
            image, camera = rotate_image(image, camera)
            if os.path.exists(matting_fname):
                matting, camera = rotate_image(matting, camera)

        # Convert to torch tensors
        image = (
            torch.from_numpy(image.astype(np.float32))
            .permute(2, 0, 1)
            .contiguous()
        )  # (3, H, W) in 0-255 range
        matting = (
            torch.from_numpy(matting.astype(np.float32))
            .unsqueeze(0)
            .contiguous()
        )  # (1, H, W) in 0-255 range
        
        intrinsics = torch.from_numpy(camera["intrinsics"].astype(np.float32))
        
        # Build 4×4 extrinsics (w2c)
        ext_3x4 = camera["extrinsics"].astype(np.float32)  # (3, 4) w2c
        ext_4x4 = np.eye(4, dtype=np.float32)
        ext_4x4[:3, :] = ext_3x4
        extrinsics = torch.from_numpy(ext_4x4)

        return {
            "image": image,
            "matting": matting,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "image_size": camera["image_size"],
        }

    # -------------------------------------------------------------------------
    # Core dataset interface
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.split_list)

    def __getitem__(self, index: int):
        subject, sequence, frame = self.split_list[index]

        # Read all calibration files for this frame
        calib_dir = self._calibration_dir(subject, sequence)
        calib_fnames = sorted(glob.glob(os.path.join(calib_dir, "*.tka")))

        all_images = []
        all_mattings = []
        all_intrinsics = []
        all_extrinsics = []
        image_size = None

        for calib_fname in calib_fnames:
            view_name = get_filename(calib_fname)

            # Filter by camera type
            is_stereo = "_A" in view_name or "_B" in view_name
            is_color = "_C" in view_name
            if is_stereo and not self.load_stereo:
                continue
            if is_color and not self.load_color:
                continue

            result = self._read_img_with_camera(
                subject, sequence, frame, calib_fname, to_meters=False
            )
            if result is None:
                continue

            all_images.append(result["image"])
            # matting
            all_mattings.append(result["matting"])
            all_intrinsics.append(result["intrinsics"])
            all_extrinsics.append(result["extrinsics"])
            if image_size is None:
                image_size = result["image_size"]

        if len(all_images) < 2:
            # Not enough views — return None (will be skipped by collate)
            return self.__getitem__((index + 1) % len(self))

        # Stack into tensors
        images = torch.stack(all_images)  # (V, 3, H, W) in 0-255
        images = images / 255.0  # normalise to 0-1
        mattings = torch.stack(all_mattings)  # (V, H, W)
        mattings = mattings / 255.0  # normalise to 0-1

        # binary matting
        mattings = (mattings > 0.5).float()
        images = images * mattings

        intrinsics_raw = torch.stack(all_intrinsics)  # (V, 3, 3) pixel-space
        extrinsics_w2c = torch.stack(all_extrinsics)  # (V, 4, 4) world-to-camera

        # Convert extrinsics: w2c → c2w (pipeline convention)
        extrinsics = torch.inverse(extrinsics_w2c)  # (V, 4, 4) camera-to-world

        # Normalise intrinsics: pixel-space → normalised (divide by image dims)
        h, w = image_size[0], image_size[1]
        intrinsics = intrinsics_raw.clone()
        intrinsics[:, 0, 0] /= w  # fx
        intrinsics[:, 0, 2] /= w  # cx
        intrinsics[:, 1, 1] /= h  # fy
        intrinsics[:, 1, 2] /= h  # cy

        scene = f"{subject}/{sequence}/{frame}"

        # --- View sampling ---
        try:
            context_indices, target_indices, overlap = self.view_sampler.sample(
                scene,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            # Not enough frames for the sampler; skip to next
            return self.__getitem__((index + 1) % len(self))

        # Skip if FOV too wide
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            return self.__getitem__((index + 1) % len(self))

        # Gather context / target images
        context_images = images[context_indices]
        target_images = images[target_indices]

        # Skip if shapes are wrong
        context_image_invalid = context_images.shape[1:] != (
            3,
            *self.cfg.original_image_shape,
        )
        target_image_invalid = target_images.shape[1:] != (
            3,
            *self.cfg.original_image_shape,
        )
        context_matting_invalid = mattings[context_indices].shape[1:] != (1,
            *self.cfg.original_image_shape,
        )
        target_matting_invalid = mattings[target_indices].shape[1:] != (
            1,  
            *self.cfg.original_image_shape,
        )

        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid or context_matting_invalid or target_matting_invalid):
            if context_image_invalid or target_image_invalid:
                print(
                    f"Skipped bad example {scene}. Context shape was "
                    f"{context_images.shape} and target shape was "
                    f"{target_images.shape}."
                )
            elif context_matting_invalid or target_matting_invalid:
                print(
                    f"Skipped bad example {scene}. Context matting shape was "
                    f"{mattings[context_indices].shape} and target matting shape was "
                    f"{mattings[target_indices].shape}."
                )
            return self.__getitem__((index + 1) % len(self))

        # --- Baseline normalisation ---
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                return self.__getitem__((index + 1) % len(self))
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        # --- Relative pose ---
        if self.cfg.relative_pose:
            extrinsics = camera_normalization(
                extrinsics[context_indices][0:1], extrinsics
            )

        # --- Build output dict ---
        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "overlap": overlap,
                # "matting": mattings[context_indices],
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,    
                "matting": mattings[target_indices],
            },
            "scene": scene,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        return apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
