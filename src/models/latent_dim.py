import torch
import torch.nn as nn
import pytorch_lightning as pl

class LatentProjection(pl.LightningModule):
    def __init__(self, eeg_latent_dim, mnist_latent_dim, hidden_dims=[256, 128]):
        super(LatentProjection, self).__init__()
        
        layers = []
        input_dim = eeg_latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, mnist_latent_dim))
        
        self.projection = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.projection(x)

    def training_step(self, batch, batch_idx):
        eeg_latent, mnist_latent, _ = batch
        projected = self(eeg_latent)
        loss = self.criterion(projected, mnist_latent)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)        
        return loss

    def validation_step(self, batch, batch_idx):
        eeg_latent, mnist_latent, _ = batch
        projected = self(eeg_latent)
        val_loss = self.criterion(projected, mnist_latent)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
