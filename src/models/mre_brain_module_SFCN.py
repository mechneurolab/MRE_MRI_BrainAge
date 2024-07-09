# from typing import Any, List
from typing import Any, Dict, Optional, Tuple
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from src.data.mre_brain_datautils import get_bin_centers

class MREBrainLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # distributed_target: bool = False,
        bin_range: Tuple[int, int] = (0,100),
        bin_step: int = 1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.HuberLoss()
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.bin_centers = get_bin_centers(self.hparams.bin_range, self.hparams.bin_step)

        # metric objects for calculating and averaging MAE across batches
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation MAE
        self.val_mae_best = MinMetric()
        self.val_loss_best = MinMetric()

    # def forward(self, x: torch.Tensor):
        # return self.net(x)
            
    def forward(self, x:torch.Tensor, x_cat:torch.Tensor):
        return self.net(x,x_cat)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_mae.reset()
        self.val_mae_best.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        x, x_cat, y, y_distribution = batch
        preds_distribution = self.forward(x, x_cat)
        preds_target = torch.matmul(preds_distribution,self.bin_centers)
        loss = self.criterion(preds_distribution, y_distribution)
        # preds = torch.argmax(logits, dim=1)
        return loss, preds_distribution, preds_target, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds_distribution, preds_target, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mae(preds_target, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds_target, "targets": targets}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `on_train_epoch_end`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds_distribution, preds_target, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_mae(preds_target, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds_target, "targets": targets}

    def on_validation_epoch_end(self):
        mae = self.val_mae.compute()  # get current val acc
        loss = self.val_loss.compute()
        self.val_mae_best(mae)  # update best so far val acc
        self.val_loss_best(loss)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds_distribution, preds_target, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_mae(preds_target, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds_target, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch: Any, batch_idx: int):
        loss, preds_distribution, preds_target, targets = self.model_step(batch)
        return loss, preds_distribution, preds_target, targets

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MREBrainLitModule(None, None, None)
