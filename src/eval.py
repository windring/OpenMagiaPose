from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    predictions = trainer.predict(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=cfg.ckpt_path)
    """
    (Pdb) predictions[0]
    (tensor([10]), tensor([0.7411]), 16.902446746826172, tensor([10]), tensor([0.7971], dtype=torch.float64))
    (Pdb) predictions[1]
    (tensor([13]), tensor([0.8721]), 7.370471954345703, tensor([12]), tensor([0.7583], dtype=torch.float64))
    (Pdb) predictions[2]
    (tensor([0]), tensor([0.7922]), 5.156755447387695, tensor([0]), tensor([0.8067], dtype=torch.float64))
    (Pdb) predictions[3]
    (tensor([8]), tensor([0.7064]), 4.777193069458008, tensor([8]), tensor([0.7378], dtype=torch.float64))
    """
    len_pred = len(predictions)
    csv_writer = open("".join(cfg.tags) + ".csv", "w")
    for i in range(len_pred):
        item = predictions[i]
        pred_label = int(item[0])
        pred_score = float(item[1])
        cost_time = float(item[2])
        gt_label = int(item[3])
        gt_score = float(item[4])
        csv_writer.write(f"{pred_label},{pred_score},{cost_time},{gt_label},{gt_score}\n")
    csv_writer.close()

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
    """
    # finetune vivit
    CUDA_VISIBLE_DEVICES=7 python src/eval.py experiment=vivit ckpt_path="/home/bob/magia_pose/logs/train/runs/2025-11-28_10-57-25/checkpoints/epoch_063_000640_weighted_loss_0.0433.ckpt"
    # dinov3_vivit
    CUDA_VISIBLE_DEVICES=7 python src/eval.py experiment=dinov3_vivit ckpt_path="/home/bob/magia_pose/logs/train/runs/2025-11-29_07-52-06/checkpoints/epoch_099_001000_weighted_loss_0.5963.ckpt"
    # dinov3_vivit_mamba
    CUDA_VISIBLE_DEVICES=7 python src/eval.py experiment=dinov3_vivit_mamba ckpt_path="/home/bob/magia_pose/logs/train/runs/2025-11-29_06-28-26/checkpoints/epoch_059_000600_weighted_loss_0.3370.ckpt"
    # dinov3_vivit_hybrid
    CUDA_VISIBLE_DEVICES=7 python src/eval.py experiment=dinov3_vivit_hybrid ckpt_path="/home/bob/magia_pose/logs/train/runs/2025-11-29_06-29-39/checkpoints/epoch_056_000570_weighted_loss_0.5404.ckpt"
    """
