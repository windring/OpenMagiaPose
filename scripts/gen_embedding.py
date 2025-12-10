import numpy as np
import json
from pathlib import Path
import rootutils
from loguru import logger
from transformers import DINOv3ViTImageProcessorFast, DINOv3ViTModel
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import argparse


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.data.video_dataset import VideoDataset

CACHE_NPY_DIR = "data/video_cache/cropped"
OUT_NPY_DIR = "data/vit7b_pca_cache"
DINOV3_PATH = "weights/dinov3-vit7b16-pretrain-lvd1689m"
LABEL2RATIOS_PATH = "data/label2ratios.json"
ANNO_JSON = "data/video_annotations.json"
BATCH_SIZE = 8
SCALER = 4
SKIP_TOKENS = 5
ORI_H = 14
ORI_W = 14


def main(num_segments, segment_idx, gpu_idx):
    label2ratios = json.load(open(LABEL2RATIOS_PATH, "r", encoding="utf-8"))
    logger.info(f"label2ratios: {label2ratios}")

    score_dict, class_dict = VideoDataset.get_score_and_class_dict(json.load(open(ANNO_JSON, "r", encoding="utf-8")))

    out_npy_dir = Path(OUT_NPY_DIR)
    out_npy_dir.mkdir(parents=True, exist_ok=True)
    npys = list(Path(CACHE_NPY_DIR).glob("*.npy"))
    npys = sorted(npys)

    num_videos_per_segment = (len(npys) + num_segments - 1) // num_segments
    batched_npys = [npys[i:i+num_videos_per_segment] for i in range(0, len(npys), num_videos_per_segment)]
    npys = batched_npys[segment_idx]

    logger.info(f"Found {len(npys)} npys")
    image_processor = DINOv3ViTImageProcessorFast.from_pretrained(DINOV3_PATH, local_files_only=True)
    image_processor.size = {"height": 224 * SCALER, "width": 224 * SCALER}
    model = DINOv3ViTModel.from_pretrained(DINOV3_PATH, local_files_only=True)
    logger.info(f"device: {gpu_idx}")
    model = model.to(f"cuda:{gpu_idx}")
    model = model.eval()
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

        inputs = image_processor(images=video_npy, return_tensors="pt")["pixel_values"] # torch.Size([949, 3, 448, 448])
        lengths = len(inputs)
        batches = [inputs[i:i+BATCH_SIZE] for i in range(0, lengths, BATCH_SIZE)]
        out_npy_list = []

        # 第一步：收集整个视频的所有embedding
        all_video_embeddings = []
        for input_batch in tqdm(batches, leave=True, desc="Collecting embeddings", position=1):
            input_batch = input_batch.to(model.device)
            with torch.no_grad():
                outputs = model(pixel_values=input_batch)
            del input_batch
            embedding = outputs.last_hidden_state.detach().cpu()
            embedding = embedding[:, SKIP_TOKENS:, :]
            
            for i in range(len(embedding)):
                item = embedding[i]
                mean = item.mean(dim=0, keepdim=True)
                center = item - mean
                all_video_embeddings.append(center)

        # 第二步：整个视频一起PCA
        all_centered = torch.cat(all_video_embeddings, dim=0)  # (N*784, 4096)
        u, s, v_global = torch.pca_lowrank(all_centered, q=3)

        # 固定符号
        for j in range(3):
            if v_global[0, j] < 0:
                v_global[:, j] = -v_global[:, j]

        # 第三步：对所有frames使用同一个v进行降维
        for center in tqdm(all_video_embeddings, desc="Reducing dimensions", position=2):
            reduced = center @ v_global[:, :3]
            img_hwc = reduced.view(ORI_H*SCALER, ORI_W*SCALER, 3)
            img_chw = img_hwc.permute(2, 0, 1)
            
            img_npy = img_chw.cpu().numpy().astype(np.float32)
            out_npy_list.append(img_npy)

        out_npy = np.stack(out_npy_list, axis=0)
        np.save(out_npy_dir / f"{npy.stem}.npy", out_npy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_segments", type=int, default=6)
    parser.add_argument("--segment_idx", type=int, default=0)
    parser.add_argument("--gpu_idx", type=int, default=0)
    args = parser.parse_args()
    main(num_segments=args.num_segments, segment_idx=args.segment_idx, gpu_idx=args.gpu_idx)
    """
    parallel -j 6 --ungroup --tagstring '[seg={1} gpu={2}]' \
    'python -u scripts/gen_embedding.py --num_segments 6 --segment_idx {1} --gpu_idx {2}' \
    ::: 0 1 2 3 4 5 :::+ 1 2 3 5 6 7
    """
