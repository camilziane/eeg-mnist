from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


def log_train_loss_accuracy(model: pl.LightningModule, loss, accuracy):
    model.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    model.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)


def log_val_loss_accuracy(
    model: pl.LightningModule, loss, accuracy, learning_rate=True
):
    model.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    model.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
    if learning_rate:
        model.log(
            "learning_rate",
            model.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )


class LossLogger(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"].item()
        accuracy = outputs["accuracy"].item()
        log_train_loss_accuracy(pl_module, loss, accuracy)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"]
        accuracy = outputs["accuracy"]
        log_val_loss_accuracy(pl_module, loss, accuracy, learning_rate=False)


class GradientLogger(Callback):
    def on_after_backward(self, trainer, pl_module):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        pl_module.log(
            "gradient_norm", total_norm, on_step=False, on_epoch=True, prog_bar=True
        )
