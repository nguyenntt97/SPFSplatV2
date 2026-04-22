"""Visualize FaMoS cameras + RGB images using Rerun, loading data via Hydra config.

Usage:
    python playground/camera_pose_check.py [--stage train] [--max_samples 5]

This mirrors the config resolution of:
    python -m src.main +experiment=spfsplatv2/famos
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import load_typed_config
from src.dataset import get_dataset, DatasetCfgWrapper
from src.dataset.dataset_famos import DatasetFaMoSCfgWrapper
from src.misc.step_tracker import StepTracker

try:
    import rerun as rr
except ImportError:
    print("Please install rerun-sdk:  pip install rerun-sdk")
    sys.exit(1)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_dataset_from_hydra(stage: str):
    """Use Hydra compose API to build a DatasetFaMOS with the same config as training."""
    config_dir = str(PROJECT_ROOT / "config")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="main",
            overrides=["+experiment=spfsplatv2/famos"],
        )

    # Extract the dataset sub-config (same path the training pipeline uses)
    dataset_dict = OmegaConf.to_container(cfg.dataset, resolve=True)

    # Build typed wrapper(s) — mirrors src.config.separate_dataset_cfg_wrappers
    from dataclasses import dataclass
    from dacite import Config, from_dict

    @dataclass
    class Dummy:
        dummy: DatasetCfgWrapper

    dataset_cfg_wrappers = []
    for k, v in dataset_dict.items():
        wrapper = from_dict(
            Dummy,
            {"dummy": {k: v}},
            config=Config(type_hooks={Path: Path}),
        ).dummy
        dataset_cfg_wrappers.append(wrapper)

    # Instantiate dataset(s) for the requested stage
    step_tracker = StepTracker()
    datasets = get_dataset(dataset_cfg_wrappers, stage, step_tracker)
    return datasets


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import glob
    import os
    from src.utils.camera import load_mpi_camera, rotate_image
    from src.utils.utils import get_filename

    parser = argparse.ArgumentParser(
        description="Visualize FaMoS cameras + RGB with Rerun (Hydra config)"
    )
    parser.add_argument(
        "--stage", type=str, default="train", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--save", type=str, default="famos_camera_check.rrd",
        help="Save recording to .rrd file (default: famos_camera_check.rrd)",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start a web viewer (access via browser after SSH port-forwarding)",
    )
    parser.add_argument(
        "--camera_size", type=float, default=10.0,
        help="Size of the camera frustum in the 3D viewer (default: 10.0)",
    )
    args = parser.parse_args()

    # Build dataset via Hydra config (same as training)
    datasets = build_dataset_from_hydra(args.stage)
    dataset = datasets[0]  # FaMoS experiment has one dataset
    print(f"Stage '{args.stage}' — dataset has {len(dataset)} samples.")

    # ── Load views from the first sample ─────────────────────────────────────
    views = []
    
    if hasattr(dataset, "split_list"):
        # FaMoS specific: Bypass view sampler to load ALL views
        subject, sequence, frame = dataset.split_list[0]
        scene_name = f"{subject}/{sequence}/{frame}"
        print(f"Visualizing all views for: {scene_name}")

        calib_dir = dataset._calibration_dir(subject, sequence)
        calib_fnames = sorted(glob.glob(os.path.join(calib_dir, "*.tka")))

        for calib_fname in calib_fnames:
            view_name = get_filename(calib_fname)
            is_stereo = "_A" in view_name or "_B" in view_name
            is_color = "_C" in view_name
            if is_stereo and not dataset.load_stereo: continue
            if is_color and not dataset.load_color: continue

            result = dataset._read_img_with_camera(subject, sequence, frame, calib_fname, to_meters=True)
            if result: views.append({"name": view_name, **result})
    elif hasattr(dataset, "chunks"):
        # RE10K / DL3DV specific: Bypass view sampler to load first 10 frames
        chunk_path = dataset.chunks[0]
        # RE10K data is saved as a list of dicts in .torch files
        chunk = torch.load(chunk_path, weights_only=True)
        example = chunk[3]
        scene_name = example["key"]
        
        num_frames = min(60, len(example["images"]))
        print(f"Visualizing {num_frames} consecutive frames for RE10k sequence: {scene_name}")
        
        # Convert raw poses
        extrinsics, intrinsics = dataset.convert_poses(example["cameras"])
        
        # Load up to 10 images
        raw_images = [example["images"][i] for i in range(num_frames)]
        images = dataset.convert_images(raw_images)  # (N, 3, H, W)
        
        for v_idx in range(num_frames):
            img = (images[v_idx] * 255).clamp(0, 255)
            h, w = img.shape[1], img.shape[2]
            
            # Un-normalise intrinsics back to pixel-space
            K = intrinsics[v_idx].clone()
            K[0, 0] *= w; K[0, 2] *= w
            K[1, 1] *= h; K[1, 2] *= h
            
            # Convert w2c to c2w if necessary. Actually RE10k convert_poses returns w2c.inverse() which IS c2w.
            # But the rest of the script expects w2c so it can invert it back to c2w. Let's just pass w2c.
            # wait, convert_poses does: return w2c.inverse(), intrinsics
            # so it returns c2w. We need to invert it back to w2c for the script.
            c2w = extrinsics[v_idx]
            ext_w2c = torch.inverse(c2w)
            
            views.append({
                "name": f"frame_{v_idx:04d}",
                "image": img,
                "extrinsics": ext_w2c,
                "intrinsics": K,
                "image_size": (h, w)
            })
    else:
        raise NotImplementedError("This script only supports FaMoS and RE10K/DL3DV datasets so far.")

    print(f"Found {len(views)} views.")

    # ── Init Rerun ───────────────────────────────────────────────────────────
    rr.init("camera_check")
    if args.serve:
        server_uri = rr.serve_grpc()
        rr.serve_web_viewer(open_browser=False)
        
        # Determine the server's IP address to construct a valid URL for VPN access
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            server_ip = s.getsockname()[0]
            s.close()
        except Exception:
            server_ip = "<SERVER_IP>"
            
        from urllib.parse import quote
        # Replace 127.0.0.1 with the actual server IP so the browser fetches from the server
        grpc_uri = f"rerun+http://{server_ip}:9876/proxy"
        viewer_url = f"http://{server_ip}:9090?url={quote(grpc_uri, safe='')}"
        
        print("\n" + "="*80)
        print("Rerun Server is running!")
        print(f"\n[Option 1] Web Viewer (Browser via VPN):")
        print(f"  {viewer_url}")
        print(f"\n[Option 2] Native Desktop Viewer (Run this in your local terminal):")
        print(f"  rerun {grpc_uri}")
        print("\n" + "="*80 + "\n")

    # Log World Coordinate Origin (X=Red, Y=Green, Z=Blue)
    rr.log(
        "world/origin",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[args.camera_size, 0, 0], [0, args.camera_size, 0], [0, 0, args.camera_size]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["X", "Y", "Z"],
        )
    )

    for v_idx, v in enumerate(views):
        entity = f"world/{v['name']}"

        img = v["image"]              # (3, H, W) float 0-255
        ext_w2c = v["extrinsics"]     # (4, 4) world-to-camera
        K = v["intrinsics"]           # (3, 3) pixel-space

        # c2w = inverse(w2c)
        c2w = torch.inverse(ext_w2c).numpy()

        h, w = v["image_size"]
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()

        # Log camera transform
        rr.log(
            entity,
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
                from_parent=False,
            ),
        )
        # Log pinhole camera
        rr.log(
            entity,
            rr.Pinhole(
                resolution=[w, h],
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                image_plane_distance=args.camera_size,
            ),
        )

        # Log RGB image (convert to uint8, HWC)
        vis_image = img.permute(1, 2, 0).numpy().astype(np.uint8)
        # Downsample for viewer performance
        vis_image = cv2.resize(
            vis_image, (w // 2, h // 2), interpolation=cv2.INTER_AREA
        )
        rr.log(f"{entity}/rgb", rr.Image(vis_image))
        print(f"  [{v_idx:2d}] {v['name']}  ({w}×{h})")

    print(f"\n✓ Logged all {len(views)} views for {scene_name}")

    if args.serve:
        print("\nServing. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        rr.save(args.save)
        print(f"\nSaved recording to: {args.save}")
        print("Open locally with:  rerun famos_camera_check.rrd")


if __name__ == "__main__":
    main()
