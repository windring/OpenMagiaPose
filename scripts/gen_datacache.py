from tqdm import tqdm
import albumentations as A
from transformers import VivitImageProcessor
from loguru import logger
import rootutils
from ultralytics import YOLO
from pathlib import Path
from typing import List
import argparse
import json


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.data.video_dataset import VideoDataset, create_video_cache


TRAIN_TXT = "data/train.txt"
VAL_TXT = "data/val.txt"
TEST_TXT = "data/test.txt"
ANNO_JSON = "data/video_annotations.json"
VIVIT_CKPT = "models/vivit-b-16x2-kinetics400"
YOLOV8S_CKPT = "models/yolov8s/yolov8s.pt"
CACHE_DIR = "data/video_cache"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_segments", type=int, default=20)
    parser.add_argument("--segment_idx", type=int, default=0)
    parser.add_argument("--gpu_idx", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(YOLOV8S_CKPT)

    def get_video_paths(filelist_path: str) -> List[Path]:
        with open(filelist_path, "r") as f:
            video_paths = [Path(x.strip()) for x in f.readlines() if len(x.strip()) > 0]
        return video_paths
    
    train_video_paths = get_video_paths(TRAIN_TXT)
    val_video_paths = get_video_paths(VAL_TXT)
    test_video_paths = get_video_paths(TEST_TXT)
    all_video_paths = train_video_paths + val_video_paths + test_video_paths
    all_video_paths = sorted(all_video_paths)

    num_segments = args.num_segments
    num_videos_per_segment = (len(all_video_paths) + num_segments - 1) // num_segments
    batched_all_video_paths = [all_video_paths[i:i+num_videos_per_segment] for i in range(0, len(all_video_paths), num_videos_per_segment)]
    current_video_paths = batched_all_video_paths[args.segment_idx]
    
    create_video_cache(model, args.gpu_idx, current_video_paths, CACHE_DIR)


def test_video_cache():
    # train_transforms = A.Compose(
    #     [
    #         A.RandomBrightnessContrast(p=0.2),
    #         A.RandomFog(p=0.2),
    #         A.RandomShadow(p=0.2),
    #         A.RandomSunFlare(p=0.2),
    #     ]
    # )


    image_processor = VivitImageProcessor.from_pretrained(
        VIVIT_CKPT
    )

    score_json_obj = json.load(
        open(ANNO_JSON, "r", encoding="utf-8")
    )

    score_dict, class_dict = VideoDataset.get_score_and_class_dict(score_json_obj)

    train_filelist = VideoDataset.get_video_list(TRAIN_TXT)
    val_filelist = VideoDataset.get_video_list(VAL_TXT)
    test_filelist = VideoDataset.get_video_list(TEST_TXT)

    logger.info(f"score_dict: {score_dict}")
    logger.info(f"class_dict: {class_dict}")

    train_dataset = VideoDataset(
        train_filelist,
        score_dict=score_dict,
        class_dict=class_dict,
        image_processor=image_processor,
        # image_transform=train_transforms,
        cache_dir=CACHE_DIR,
    )
    logger.info(train_dataset)
    for item in tqdm(iter(train_dataset)):
        pass

    val_dataset = VideoDataset(
        val_filelist,
        score_dict=score_dict,
        class_dict=class_dict,
        image_processor=image_processor,
        cache_dir=CACHE_DIR,
    )
    logger.info(val_dataset)
    for item in tqdm(iter(val_dataset)):
        pass

    test_dataset = VideoDataset(
        test_filelist,
        score_dict=score_dict,
        class_dict=class_dict,
        image_processor=image_processor,
        cache_dir=CACHE_DIR,
    )
    logger.info(test_dataset)
    for item in tqdm(iter(test_dataset)):
        pass
