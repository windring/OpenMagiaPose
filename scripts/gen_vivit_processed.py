import numpy as np
import json
from pathlib import Path
import rootutils
from loguru import logger
from tqdm import tqdm


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.video_dataset import VideoDataset


CACHE_NPY_DIR = "data/video_cache/cropped"
OUT_NPY_DIR = "data/vivit_processed_cache"
LABEL2RATIOS_PATH = "data/label2ratios.json"
ANNO_JSON = "data/video_annotations.json"



def main():
    label2ratios = json.load(open(LABEL2RATIOS_PATH, "r", encoding="utf-8"))
    logger.info(f"label2ratios: {label2ratios}")

    score_dict, class_dict = VideoDataset.get_score_and_class_dict(json.load(open(ANNO_JSON, "r", encoding="utf-8")))

    out_npy_dir = Path(OUT_NPY_DIR)
    out_npy_dir.mkdir(parents=True, exist_ok=True)
    npys = list(Path(CACHE_NPY_DIR).glob("*.npy"))
    npys = sorted(npys)

    logger.info(f"Found {len(npys)} npys")

    for npy in tqdm(npys, leave=True, desc="Processing npy", position=0):
        video_name = f"{npy.stem}.mp4"
        label = class_dict[video_name]
        score = score_dict[video_name]
        ratios = label2ratios[str(label)]
        logger.info(f"video_name: {video_name}, label: {label}, score: {score}, ratios: {ratios}")

        video_npy = np.load(npy) # (949, 224, 224, 3)
        logger.info(f"origin video_npy.shape: {video_npy.shape}")
        video_npy = video_npy[list(map(int, np.array(ratios) * (video_npy.shape[0] - 1))), :, :, :]
        logger.info(f"select video_npy.shape: {video_npy.shape}")

        np.save(out_npy_dir / f"{npy.stem}.npy", video_npy)

if __name__ == "__main__":
    main()
