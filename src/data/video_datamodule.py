import json
import logging
import sys
from typing import Any, Dict, Optional

import hydra
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from transformers import VivitImageProcessor
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from src.data.video_dataset import VideoDataset
from src.utils.pylogger import RankedLogger


logger = RankedLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# 自定义函数，转换 BGR -> RGB
def bgr_to_rgb(image: np.ndarray):
    if image.ndim != 4:
        raise ValueError(f"Expected 4D array, got {image.ndim}D array")
    return image[:, :, :, ::-1]  # 反转通道顺序


# 自定义函数，转换 RGB -> BGR
def rgb_to_bgr(image: np.ndarray):
    if image.ndim != 4:
        raise ValueError(f"Expected 4D array, got {image.ndim}D array")
    return image[:, :, :, ::-1]  # 反转通道顺序

class RandomColorMap(ImageOnlyTransform):
    """随机色表映射：负片、通道交换、颜色反转等"""
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self, img, **params):
        mode = np.random.randint(0, 10)
        
        if mode == 0:  # 负片效果
            img = 255 - img
        elif mode == 1:  # RGB -> BGR
            img = img[:, :, ::-1]
        elif mode == 2:  # RGB -> GRB
            img = img[:, :, [1, 0, 2]]
        elif mode == 3:  # RGB -> RBG
            img = img[:, :, [0, 2, 1]]
        elif mode == 4:  # RGB -> BRG
            img = img[:, :, [2, 0, 1]]
        elif mode == 5:  # RGB -> GBR
            img = img[:, :, [1, 2, 0]]
        elif mode == 6:  # 单通道反转（随机选择）
            channel = np.random.randint(0, 3)
            img[:, :, channel] = 255 - img[:, :, channel]
        elif mode == 7:  # 负片 + BGR
            img = 255 - img[:, :, ::-1]
        elif mode == 8:  # 通道循环移位
            shift = np.random.randint(1, 3)
            img = np.roll(img, shift, axis=2)
        # mode == 9: 保持原样
        
        return img
    
    def get_transform_init_args_names(self):
        return ("p",)


class VideoDataModule(LightningDataModule):

    def __init__(
        self,
        batch_size: int = 64,
        num_classes: int = 15,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_filelist_path: str = "",
        val_filelist_path: str = "",
        test_filelist_path: str = "",
        anno_json_fn: str = "",
        cache_dir: str = "",
        image_processor_path: str = "models/vivit-b-16x2-kinetics400",
        use_vivit_processor: bool = True,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        assert self.hparams.num_classes == 15

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            train_transforms = A.Compose(
                [
                    A.OneOf(
                        [
                            A.Compose(
                                [
                                    A.RandomFog(p=0.2),
                                    A.RandomShadow(p=0.2),
                                    A.RandomSunFlare(p=0.2),
                                    A.GaussNoise(),
                                    A.OneOf(
                                        [
                                            A.MotionBlur(p=0.2),
                                            A.MedianBlur(blur_limit=3, p=0.1),
                                            A.Blur(blur_limit=3, p=0.1),
                                        ],
                                        p=0.2,
                                    ),
                                    # A.ShiftScaleRotate(
                                    #     shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
                                    # ),
                                    A.OneOf(
                                        [
                                            A.OpticalDistortion(p=0.3),
                                            A.GridDistortion(p=0.1),
                                        ],
                                        p=0.2,
                                    ),
                                    A.OneOf(
                                        [
                                            A.CLAHE(clip_limit=2),
                                            A.RandomBrightnessContrast(),
                                        ],
                                        p=0.3,
                                    ),
                                    A.HueSaturationValue(p=0.3),
                                    A.Normalize(),
                                ]
                            ),
                            A.Normalize(),
                        ]
                    )
                ]
            )
            val_transforms = A.Compose(
                [
                    A.Normalize(),
                ]
            )

            if self.hparams.use_vivit_processor:
                logger.warning("Using Vivit image processor")
                image_processor = VivitImageProcessor.from_pretrained(
                    self.hparams.image_processor_path
                )
            else:
                logger.warning("Using None image processor")
                image_processor = None
                train_transforms = A.Compose([
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5
                    ),
                    A.ColorJitter(p=0.5),
                ])
                val_transforms = None

            anno_json_obj = json.load(
                open(self.hparams.anno_json_fn, "r", encoding="utf-8")
            )

            score_dict, class_dict = VideoDataset.get_score_and_class_dict(
                anno_json_obj
            )

            train_filelist = VideoDataset.get_video_list(self.hparams.train_filelist_path)
            val_filelist = VideoDataset.get_video_list(self.hparams.val_filelist_path)
            test_filelist = VideoDataset.get_video_list(self.hparams.test_filelist_path)

            trainset = VideoDataset(
                train_filelist,
                cache_dir=self.hparams.cache_dir,
                score_dict=score_dict,
                class_dict=class_dict,
                image_processor=image_processor,
                image_transform=train_transforms,
                use_vivit_processor=self.hparams.use_vivit_processor,
                random_swap_rgb=True,
            )
            logger.info(f"trainset: {trainset}")
            valset = VideoDataset(
                val_filelist,
                cache_dir=self.hparams.cache_dir,
                score_dict=score_dict,
                class_dict=class_dict,
                image_processor=image_processor,
                image_transform=val_transforms,
                use_vivit_processor=self.hparams.use_vivit_processor,
            )
            logger.info(f"valset: {valset}")
            testset = VideoDataset(
                test_filelist,
                cache_dir=self.hparams.cache_dir,
                score_dict=score_dict,
                class_dict=class_dict,
                image_processor=image_processor,
                image_transform=val_transforms,
                use_vivit_processor=self.hparams.use_vivit_processor,
            )
            logger.info(f"testset: {testset}")

            self.data_train = trainset
            self.data_val = valset
            self.data_test = testset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    logger.info(dm)
    logger.info(f"sample: {next(iter(dm.train_dataloader()))}")


if __name__ == "__main__":
    main()
