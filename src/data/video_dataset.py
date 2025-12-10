import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import albumentations as A
import cv2
import numpy as np
from moviepy import ImageSequenceClip, VideoFileClip
from torch.utils.data import Dataset
from transformers import VivitImageProcessor
from tqdm import tqdm

from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


BOOL_MAGIA_DEBUG = os.environ.get("MAGIA_DEBUG", "False").lower() == "true"
INT_YOLO_BATCH_SIZE = 256

class VideoDataset(Dataset):

    def __init__(
        self,
        filelist: List[str],
        score_dict: Dict[str, float] = None,
        class_dict: Dict[str, int] = None,
        image_processor: VivitImageProcessor = None,
        image_transform: A.Compose = None,
        cache_dir: str = None,
        use_vivit_processor: bool = True,
        random_swap_rgb: bool = False,
    ):
        logger.warn(f"image_processor: {image_processor}")
        logger.warn(f"image_transform: {image_transform}")
        logger.warn(f"use_vivit_processor: {use_vivit_processor}")
        logger.warn(f"random_swap_rgb: {random_swap_rgb}")

        self.filelist = filelist

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is None or not self.cache_dir.exists():
            raise ValueError(f"Cache directory {self.cache_dir} does not exist")

        self.score_dict = score_dict
        self.class_dict = class_dict
        self.image_processor = image_processor
        self.image_transform = image_transform
        self.use_vivit_processor = use_vivit_processor
        self.random_swap_rgb = random_swap_rgb if not use_vivit_processor else False

    @staticmethod
    def get_score_and_class_dict(anno_list: List[Any]):
        class_dict = dict()
        score_dict = dict()
        for item in anno_list:
            logger.info(f"item: {item}")

            if item["videoName"] in score_dict:
                raise ValueError(f"Duplicate video name: {item['videoName']}")
            if item["videoName"] in class_dict:
                raise ValueError(f"Duplicate video name: {item['videoName']}")

            cur_score = None
            cur_label = None

            if "label" in item and item["label"] is not None:
                cur_label = int(item["label"])
            else:
                logger.warning(f"label is not found for {item['videoName']}")
                continue

            if "standardSocre" in item and item["standardSocre"] is not None:
                cur_score = float(item["standardSocre"])
            elif "standardSocre" in item and item["standardSocre"] is None and cur_label == 14:
                cur_score = 0.0
            else:
                logger.warning(f"standardSocre is not found for {item['videoName']}")
                continue       

            score_dict[item["videoName"]] = cur_score
            class_dict[item["videoName"]] = cur_label

        return score_dict, class_dict

    @staticmethod
    def get_video_list(pwd_txt: str) -> List[Path]:
        with open(pwd_txt, "r", encoding="utf-8") as f:
            video_paths = f.readlines()
        video_paths = [Path(x.strip()) for x in video_paths if len(x.strip()) > 0]
        return video_paths

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        """
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        """
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        video_path = self.filelist[index]
        logger.debug(f"Video path: {video_path}")

        label = self.class_dict[video_path.name]
        logger.debug(f"Label: {label}")

        score = self.score_dict[video_path.name]
        logger.debug(f"Score: {score}")

        cache_path = self.cache_dir / f"{video_path.stem}.npy"
        logger.debug(f"Cache path: {cache_path}")
        if os.path.exists(cache_path):
            video_npy = np.load(cache_path)
        else:
            raise FileNotFoundError(f"Cache file not found for {video_path}")

        # if self.use_vivit_processor and self.image_processor:
        #     video_npy = self.image_processor(images=video_npy, return_tensors="pt")["pixel_values"]
        # else:
        #     video_npy_max = video_npy.max()
        #     video_npy_min = video_npy.min()
        #     video_npy = (video_npy - video_npy_min) / (video_npy_max - video_npy_min)

        if self.use_vivit_processor and self.image_transform:
            video_npy = self.image_transform(images=video_npy)["images"]
            # 32, 224, 224, 3
        
        if not self.use_vivit_processor and self.image_transform:
            # images = []
            # for i in range(video_npy.shape[0]):
            #     image = video_npy[i].transpose(1, 2, 0)
            #     image = self.image_transform(image=image)["image"]
            #     images.append(image)
            # video_npy = np.stack(images, axis=0)
            # logger.info(f"video_npy.shape: {video_npy.shape}")
            video_npy = video_npy.transpose(0, 2, 3, 1)
            video_npy = self.image_transform(images=video_npy)["images"]
            video_npy = video_npy.transpose(0, 3, 1, 2)
            # logger.info(f"video_npy.shape: {video_npy.shape}")

        if BOOL_MAGIA_DEBUG:
            # save tensor as images
            import matplotlib.pyplot as plt
            import time

            os.makedirs("debug_output", exist_ok=True)
            time_str = str(time.time())
            for i in range(video_npy.shape[0]):
                plt.imshow(video_npy[i])
                plt.savefig(f"debug_output/{video_path.name}_frame_{i}_{time_str}.png")

        if self.random_swap_rgb:
            new_index = np.random.permutation(3)
            # logger.info(f"new_index: {new_index}, video_npy.shape: {video_npy.shape}")
            video_npy = video_npy[:, new_index, :, :]

        if self.use_vivit_processor:
            return video_npy.transpose(0, 3, 1, 2).astype(np.float32), int(label), float(score)
        else:
            return video_npy.astype(np.float32), int(label), float(score)


def create_video_cache(model, gpu_idx: int, video_paths: List[Path], cache_dir: str):
    """预先为所有视频创建缓存，使用moviepy提取人体并保存"""
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "cropped"), exist_ok=True)

    # 加载YOLO模型用于人体检测
    for video_path in tqdm(video_paths, desc="处理视频"):
        try:
            cache_path_cropped = os.path.join(
                cache_dir, "cropped", f"{video_path.stem}.npy"
            )
            if os.path.exists(cache_path_cropped):
                logger.warning(f"Cache file already exists for {video_path}")

            # 使用moviepy加载视频
            video = VideoFileClip(str(video_path))
            results = model(
                video_path,
                batch=INT_YOLO_BATCH_SIZE,
                classes=[0],
                verbose=False,
                device=f"cuda:{gpu_idx}",
            )

            # 第一遍遍历找到最大最小边界
            min_x1, min_y1 = float("inf"), float("inf")
            max_x2, max_y2 = float("-inf"), float("-inf")
            found_person = False

            # 处理每一批的结果
            for result in results:
                if len(result.boxes) > 0:
                    boxes = result.boxes
                    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (
                        boxes.xyxy[:, 3] - boxes.xyxy[:, 1]
                    )
                    max_idx = areas.argmax()
                    box = boxes.xyxy[max_idx]
                    x1, y1, x2, y2 = map(int, box)
                    min_x1 = min(min_x1, x1)
                    min_y1 = min(min_y1, y1)
                    max_x2 = max(max_x2, x2)
                    max_y2 = max(max_y2, y2)
                    found_person = True

            if not found_person:
                logger.info(f"视频 {video_path} 没有检测到人")
                min_x1, min_y1 = 0, 0
                max_x2, max_y2 = video.size[0], video.size[1]

            # 计算宽高和padding
            w = max_x2 - min_x1
            h = max_y2 - min_y1
            if h > w:
                new_h = h
                new_w = h
                pad_left = (new_w - w) // 2
                pad_right = new_w - w - pad_left
                crop_x1 = min_x1 - pad_left
                crop_x2 = max_x2 + pad_right
                crop_y1 = min_y1
                crop_y2 = max_y2
            else:
                new_w = w
                new_h = w
                pad_top = (new_h - h) // 2
                pad_bottom = new_h - h - pad_top
                crop_x1 = min_x1
                crop_x2 = max_x2
                crop_y1 = min_y1 - pad_top
                crop_y2 = max_y2 + pad_bottom

            # 第二遍遍历进行裁剪
            frames_cropped = []
            for frame in video.iter_frames():
                # 使用固定的裁剪框
                cropped = frame[
                    max(0, crop_y1) : min(frame.shape[0], crop_y2),
                    max(0, crop_x1) : min(frame.shape[1], crop_x2),
                ]

                # 处理边界情况的padding
                if cropped.shape[0] < new_h:
                    cropped = cv2.copyMakeBorder(
                        cropped, new_h - cropped.shape[0], 0, 0, 0, cv2.BORDER_REPLICATE
                    )
                else:
                    cropped = cropped[:new_h, :]

                if cropped.shape[1] < new_w:
                    cropped = cv2.copyMakeBorder(
                        cropped, 0, 0, 0, new_w - cropped.shape[1], cv2.BORDER_REPLICATE
                    )
                else:
                    cropped = cropped[:, :new_w]

                # resize到目标大小
                cropped = cv2.resize(cropped, (224, 224))
                frames_cropped.append(cropped)

            # 转换为numpy数组并保存
            decoded_video_cropped = np.stack(frames_cropped)

            cropped_size_mb = decoded_video_cropped.nbytes / (1024 * 1024)
            logger.info(f"视频 {video_path} 裁剪版大小: {cropped_size_mb:.2f}MB")

            # 保存为npy格式
            np.save(cache_path_cropped, decoded_video_cropped)

            # 同时保存为mp4格式
            mp4_path = cache_path_cropped.replace(".npy", ".mp4")
            clip = ImageSequenceClip([frame for frame in decoded_video_cropped], fps=30)
            clip.write_videofile(mp4_path, codec="libx264", audio=False)
            clip.close()

            video.close()

        except Exception as e:
            logger.error(f"处理视频出错 {video_path}: {str(e)}")
            import traceback

            logger.info(traceback.format_exc())
            continue

    # # 输出缓存总大小
    # cropped_size = sum(
    #     os.path.getsize(os.path.join(cache_dir, "cropped", f))
    #     for f in os.listdir(os.path.join(cache_dir, "cropped"))
    #     if f.endswith(".npy")
    # )
    # logger.info(f"裁剪版缓存总大小: {cropped_size / (1024*1024*1024):.2f}GB")
