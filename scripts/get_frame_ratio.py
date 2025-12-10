import json
import numpy as np
from loguru import logger


def parse_timestamp(ts):
    """将 '00:00:09 0' 转换为秒数"""
    time_part, frame = ts.rsplit(' ', 1)
    h, m, s = map(int, time_part.split(':'))
    assert int(frame) == 0
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds

def get_frame_ratios(annotation):
    """计算所有关键帧在 duration 中的 ratio"""
    duration = float(annotation['videoInfo']['duration'])
    fps = float(annotation['videoInfo']['fps'])
    assert fps == 30
    
    ratios = []
    
    # firstKeyFrame
    first_kf = annotation['standardSocreInfo']['firstKeyFrame']
    ratios.append(parse_timestamp(first_kf) / duration)
    
    # referenceFrames
    for ref_frame in annotation['standardSocreInfo']['referenceFrames']:
        ratios.append(parse_timestamp(ref_frame) / duration)
    
    return ratios


# 使用示例
with open('data/video_annotations.json') as f:
    annotations = json.load(f)

def main():
    label2ratios = dict()
    for ann_idx, ann in enumerate(annotations[:14]):  # standard_0 到 standard_13
        assert int(ann["label"]) == ann_idx
        ratios = get_frame_ratios(ann)
        ratios = sorted(set(ratios))  # 去重排序
        n_original = len(ratios)
        logger.info(f"n_original: {n_original}")
        n_fill = 32 - n_original
        fill_points = np.linspace(0, 1, n_fill + 2)[1:-1].tolist()
        new_ratios = ratios + fill_points
        new_ratios = sorted(set(new_ratios))
        logger.info(f"new_ratios: {new_ratios}")
        while len(new_ratios) < 32:
            new_ratios.append(np.random.uniform(0, 1))
            new_ratios = sorted(set(new_ratios))
        assert len(new_ratios) == 32, f"len(new_ratios) != 32: {len(new_ratios)}"
        print(f"{ann_idx}: {ann['videoId']}: {[r for r in new_ratios]}")
        label2ratios[ann_idx] = list(new_ratios)
    label2ratios[14] = list(np.linspace(0, 1, 32))

    with open('data/label2ratios.json', 'w') as f:
        json.dump(label2ratios, f, indent=4)

if __name__ == "__main__":
    main()