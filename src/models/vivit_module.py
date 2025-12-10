from pathlib import Path
from typing import Any, Dict, Tuple
import time

import hydra
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MaxMetric, MeanAbsoluteError, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import VivitConfig

from src.utils.pylogger import RankedLogger
from src.models.modeling_vivit import VivitModel

logger = RankedLogger(__name__)


class LinearPooling(nn.Module):
    def __init__(self, num_tokens, hidden_size, out_dim):
        super().__init__()
        self.tokens_linear = nn.Linear(num_tokens, 1)
        self.flatten = nn.Flatten()
        self.hidden_linear = nn.Linear(hidden_size, out_dim)
    
    def forward(self, x):
        # x: (batch_size, num_tokens, hidden_size)
        x = x.transpose(1, 2)      # -> (batch_size, hidden_size, num_tokens)
        x = self.tokens_linear(x)         # -> (batch_size, hidden_size, 1)
        x = self.flatten(x)            # -> (batch_size, hidden_size)
        x = self.hidden_linear(x)      # -> (batch_size, out_dim)
        return x

class VivitLitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        num_classes: int = 15,
        cls_weight: float = 0.8,
        reg_weight: float = 1.2,
        vivit_model_pretrained: str = "models/vivit-b-16x2-kinetics400",
        use_attention: bool = True,
        use_mamba: bool = False,
        use_hybrid: bool = False,
        use_pretrained_vivit: bool = True,
    ) -> None:
        """Initialize a `VivitLitModule`.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.vivit_config = VivitConfig.from_pretrained(
            self.hparams.vivit_model_pretrained
        )

        self.vivit_config.use_attention = self.hparams.use_attention
        self.vivit_config.use_mamba = self.hparams.use_mamba
        self.vivit_config.use_hybrid = self.hparams.use_hybrid
        if not self.hparams.use_pretrained_vivit:
            self.vivit_config.image_size = 56  # 224 -> 28
            self.vivit_config.tubelet_size = (2, 4, 4)  # (2, 16, 16) -> (2, 2, 2)
            self.vivit_config.hidden_size = 512
            self.vivit_config.num_hidden_layers = 2
            self.vivit_config.num_attention_heads = 2
            self.vivit_config.intermediate_size = 2048 # 3072
            self.vivit_config.hidden_dropout_prob = 0.2
            self.vivit_config.attention_probs_dropout_prob = 0.2

        # good version
        self.vivit = VivitModel(self.vivit_config, add_pooling_layer=True)

        if self.hparams.use_pretrained_vivit:
            state_dict = torch.load(Path(self.hparams.vivit_model_pretrained) / "pytorch_model.bin")
            state = {}
            for k, v in state_dict.items():
                if k.startswith("vivit."):
                    state[k[len("vivit."):]] = v
            missing_keys, unexpected_keys = self.vivit.load_state_dict(state, strict=False)
            logger.warning(f"Missing keys: {missing_keys}")
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        self.vivit.train()

        # bad version
        # if (Path(self.hparams.vivit_model_pretrained) / "pytorch_model.bin").exists():
        #     self.vivit = VivitModel.from_pretrained(self.hparams.vivit_model_pretrained)
        # else:
        #     self.vivit = VivitModel(self.vivit_config, add_pooling_layer=True)

        # Classifier head
        if self.hparams.use_pretrained_vivit:
            self.classifier = nn.Linear(
                self.vivit_config.hidden_size, self.hparams.num_classes
            )
            self.regression = nn.Linear(self.vivit_config.hidden_size, 1)
        else:
            num_tokens = self.vivit.embeddings.patch_embeddings.num_patches + 1
            logger.info(f"num_tokens: {num_tokens}")
            self.classifier = LinearPooling(num_tokens, self.vivit_config.hidden_size, self.hparams.num_classes)
            self.regression = LinearPooling(num_tokens, self.vivit_config.hidden_size, 1)

        self.regression_fct = nn.Sigmoid()

        self.cls_weight = float(self.hparams.cls_weight)
        self.reg_weight = float(self.hparams.reg_weight)

        # loss function
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_reg = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc_cls = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.val_acc_cls = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.test_acc_cls = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )

        # 回归评估指标
        self.train_mae_reg = MeanAbsoluteError()
        self.val_mae_reg = MeanAbsoluteError()
        self.test_mae_reg = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss_cls = MeanMetric()
        self.val_loss_cls = MeanMetric()
        self.test_loss_cls = MeanMetric()

        self.train_loss_reg = MeanMetric()
        self.val_loss_reg = MeanMetric()
        self.test_loss_reg = MeanMetric()

        self.train_weighted_loss = MeanMetric()
        self.val_weighted_loss = MeanMetric()
        self.test_weighted_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_mae_best = MinMetric()
        self.val_weighted_score_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vivit(x)
        last_hidden_states = x[0]
        if self.hparams.use_pretrained_vivit:
            last_hidden_states = last_hidden_states[:, 0, :]
        return self.classifier(last_hidden_states), self.regression(last_hidden_states)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc_cls.reset()
        self.val_acc_cls.reset()
        self.test_acc_cls.reset()

        # 回归评估指标
        self.train_mae_reg.reset()
        self.val_mae_reg.reset()
        self.test_mae_reg.reset()

        # for averaging loss across batches
        self.train_loss_cls.reset()
        self.val_loss_cls.reset()
        self.test_loss_cls.reset()

        self.train_loss_reg.reset()
        self.val_loss_reg.reset()
        self.test_loss_reg.reset()

        self.train_weighted_loss.reset()
        self.val_weighted_loss.reset()
        self.test_weighted_loss.reset()

        # for tracking best so far validation accuracy
        self.val_acc_best.reset()
        self.val_mae_best.reset()
        self.val_weighted_score_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, label, score = batch
        logits_cls, logits_reg = self.forward(x.float())
        logits_cls = logits_cls.view(-1, self.hparams.num_classes)
        loss_cls = self.criterion_cls(logits_cls, label.long())

        logits_reg = self.regression_fct(logits_reg)
        logits_reg = logits_reg.view(-1)
        loss_reg = self.criterion_reg(logits_reg, score.float())

        loss = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        preds_cls = torch.argmax(logits_cls, dim=1)

        return loss_cls, loss_reg, loss, preds_cls, logits_reg, label, score
    
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        start_time = time.time()
        x, label, score = batch
        logits_cls, logits_reg = self.forward(x.float())
        logits_cls = logits_cls.view(-1, self.hparams.num_classes)
        # loss_cls = self.criterion_cls(logits_cls, label.long())

        logits_reg = self.regression_fct(logits_reg)
        logits_reg = logits_reg.view(-1)
        # loss_reg = self.criterion_reg(logits_reg, score.float())

        # loss = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        preds_cls = torch.argmax(logits_cls, dim=1)
        end_time = time.time()
        cost_time = (end_time - start_time) * 1000
        logger.info(f"cost_time: {cost_time}")

        return  preds_cls, logits_reg, cost_time, label, score

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_cls, loss_reg, loss, preds_cls, logits_reg, label, score = self.model_step(
            batch
        )

        # update and log metrics
        self.train_loss_cls(loss_cls)
        self.train_loss_reg(loss_reg)
        self.train_weighted_loss(loss)
        self.train_acc_cls(preds_cls, label)
        self.train_mae_reg(logits_reg, score)

        self.log(
            "train/loss_cls",
            self.train_loss_cls,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/loss_reg",
            self.train_loss_reg,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/weighted_loss",
            self.train_weighted_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/acc_cls",
            self.train_acc_cls,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/mae_reg",
            self.train_mae_reg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss_cls, loss_reg, loss, preds_cls, logits_reg, label, score = self.model_step(
            batch
        )

        # update and log metrics
        self.val_loss_cls(loss_cls)
        self.val_loss_reg(loss_reg)
        self.val_weighted_loss(loss)
        self.val_acc_cls(preds_cls, label)
        self.val_mae_reg(logits_reg, score)
        self.log(
            "val/loss_cls",
            self.val_loss_cls,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/loss_reg",
            self.val_loss_reg,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/weighted_loss",
            self.val_weighted_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/acc_cls", self.val_acc_cls, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/mae_reg", self.val_mae_reg, on_step=False, on_epoch=True, prog_bar=True
        )
        # self.log(
        #     "val/weighted_score",
        #     self.hparams.cls_weight * self.val_acc_cls + self.hparams.reg_weight * (1-self.val_mae_reg),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc_cls.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

        mae = self.val_mae_reg.compute()
        self.val_mae_best(mae)
        self.log(
            "val/mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True
        )

        weighted_score = self.hparams.cls_weight * acc + self.hparams.reg_weight * (1-mae)
        self.val_weighted_score_best(weighted_score)
        self.log(
            "val/weighted_score_best", self.val_weighted_score_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_cls, loss_reg, loss, preds_cls, logits_reg, label, score = self.model_step(
            batch
        )

        # update and log metrics
        self.test_loss_cls(loss_cls)
        self.test_loss_reg(loss_reg)
        self.test_weighted_loss(loss)
        self.test_acc_cls(preds_cls, label)
        self.test_mae_reg(logits_reg, score)
        self.log(
            "test/loss_cls",
            self.test_loss_cls,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/loss_reg",
            self.test_loss_reg,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/weighted_loss",
            self.test_weighted_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/acc_cls",
            self.test_acc_cls,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/mae_reg",
            self.test_mae_reg,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        # self.log(
        #     "test/weighted_score",
        #     self.hparams.cls_weight * self.test_acc_cls + self.hparams.reg_weight * (1-self.test_mae_reg),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        acc_cls = self.test_acc_cls.compute()
        self.log(
            "test/classification_accuracy", acc_cls, sync_dist=True, prog_bar=True
        )

        mae = self.test_mae_reg.compute()
        self.log(
            "test/regression_mae", mae, sync_dist=True, prog_bar=True
        )

        weighted_score = self.hparams.cls_weight * acc_cls + self.hparams.reg_weight * (1-mae)
        self.log(
            "test/weighted_score", weighted_score, sync_dist=True, prog_bar=True
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.vivit = torch.compile(self.vivit)
            self.classifier = torch.compile(self.classifier)
            self.regression = torch.compile(self.regression)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/weighted_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    logger.info(model)


if __name__ == "__main__":
    main()
