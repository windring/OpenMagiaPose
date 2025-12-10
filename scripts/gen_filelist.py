import random
from pathlib import Path
import json

from loguru import logger
import hydra
from transformers import set_seed
import rootutils


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.data.video_dataset import VideoDataset


@hydra.main(config_name="gen_filelist", config_path="../configs/data", version_base=None)
def create_filelists(cfg):
    """
    创建训练集和验证集的文件列表

    :param data_root_dir: 存放所有分类视频文件的根目录
    :param train_filelist_path: 训练集文件列表的保存路径
    :param val_filelist_path: 验证集文件列表的保存路径
    """
    set_seed(cfg.seed)

    score_json_obj = json.load(open(cfg.anno_json_fn, "r", encoding="utf-8"))
    score_dict, class_dict = VideoDataset.get_score_and_class_dict(score_json_obj)

    data_root_dir = Path(cfg.data_root_dir)
    # project_dir = Path(__file__).parent.parent

    assert data_root_dir.is_dir(), f"Root directory {data_root_dir} is not a directory"
    assert data_root_dir.exists(), f"Root directory {data_root_dir} does not exist"

    class_folders = [f for f in data_root_dir.iterdir()]

    train_filelist = []
    val_filelist = []
    test_filelist = []

    for class_folder in class_folders:
        if not class_folder.is_dir():
            continue

        video_files = [
            f for f in class_folder.iterdir() if f.suffix == ".mp4" and f.name in score_dict
        ]
        video_files = sorted(video_files)

        random.shuffle(video_files)

        num_files = len(video_files)
        num_train = int(num_files * 0.8)
        num_val = int(num_files * 0.1)

        logger.info(f"class: {class_folder.name}, num_files: {num_files}, num_train: {num_train}, num_val: {num_val}, num_test: {num_files - num_train - num_val}")

        train_filelist.extend(video_files[:num_train])
        val_filelist.extend(video_files[num_train:num_train+num_val])
        test_filelist.extend(video_files[num_train+num_val:])

    train_filelist = sorted(train_filelist)
    val_filelist = sorted(val_filelist)
    test_filelist = sorted(test_filelist)

    random.shuffle(train_filelist)
    random.shuffle(val_filelist)
    random.shuffle(test_filelist)

    with open(cfg.train_filelist_path, "w", encoding="utf-8") as f:
        for file_path in train_filelist:
            f.write(str(file_path) + "\n")

    with open(cfg.val_filelist_path, "w", encoding="utf-8") as f:
        for file_path in val_filelist:
            f.write(str(file_path) + "\n")
    
    with open(cfg.test_filelist_path, "w", encoding="utf-8") as f:
        for file_path in test_filelist:
            f.write(str(file_path) + "\n")


if __name__ == "__main__":
    create_filelists()
