"""Minimal export script to write the first RE10k sequence."""

import sys
from pathlib import Path
import torch

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.dataset import get_dataset, DatasetCfgWrapper
from src.misc.step_tracker import StepTracker

def build_dataset_from_hydra(stage: str):
    config_dir = str(PROJECT_ROOT / "config")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="main",
            overrides=["+experiment=spfsplatv2/re10k"],
        )

    dataset_dict = OmegaConf.to_container(cfg.dataset, resolve=True)

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

    step_tracker = StepTracker()
    datasets = get_dataset(dataset_cfg_wrappers, stage, step_tracker)
    return datasets

def main():
    print("Building dataset via Hydra config...")
    datasets = build_dataset_from_hydra("train")
    dataset = datasets[0]

    chunk_path = dataset.chunks[0]
    # RE10K data is saved as a list of dicts in .torch files
    chunk = torch.load(chunk_path, weights_only=True)
    example = chunk[3]
    scene_name = example["key"]

    import shutil
    import torchvision

    # Save to a cache folder in the same parent as this script
    cache_dir = Path(__file__).resolve().parent / "cache"
    
    # Remove all content inside the cache folder first
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing images for scene '{scene_name}' to {cache_dir}...")
    raw_images = example["images"]
    images = dataset.convert_images(raw_images)
    print(images.shape)

    for i, img in enumerate(images):
        save_path = cache_dir / f"{i:03d}.png"
        torchvision.utils.save_image(img, save_path)
    print("Done!")

if __name__ == "__main__":
    main()
