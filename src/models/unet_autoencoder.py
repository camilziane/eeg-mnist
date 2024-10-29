import torch
import torch.nn as nn
import pytorch_lightning as pl

class UNetAutoencoder(pl.LightningModule):
    def __init__(self, input_channels=5, sequence_length=256, latent_dim=64, dropout_rate=0.2, mask_ratio=0.15):
        super(UNetAutoencoder, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.mask_ratio = mask_ratio

        # Encoder
        self.enc1 = self.conv_block(input_channels, 32, dropout_rate)
        self.enc2 = self.conv_block(32, 64, dropout_rate)
        self.enc3 = self.conv_block(64, 128, dropout_rate)
        self.enc4 = self.conv_block(128, 256, dropout_rate)

        # Calculate the size of the encoder output
        self.encoder_output_size = sequence_length // 16 * 256

        # Latent space
        self.fc1 = nn.Sequential(
            nn.Linear(self.encoder_output_size, latent_dim),
            nn.Dropout(dropout_rate)
        )
        self.fc2 = nn.Linear(latent_dim, self.encoder_output_size)

        # Decoder
        self.dec4 = self.conv_block(256, 128, dropout_rate, transpose=True)
        self.dec3 = self.conv_block(256, 64, dropout_rate, transpose=True)
        self.dec2 = self.conv_block(128, 32, dropout_rate, transpose=True)
        self.dec1 = nn.ConvTranspose1d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.criterion = nn.MSELoss()

    def conv_block(self, in_channels, out_channels, dropout_rate, transpose=False):
        if not transpose:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Flatten
        x = e4.view(e4.size(0), -1)

        # Latent space
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape for decoder
        x = x.view(x.size(0), 256, -1)

        # Decoder with skip connections
        x = self.dec4(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return x

    def get_encoder(self):
        return nn.Sequential(
            self.enc1, self.enc2, self.enc3, self.enc4,
            nn.Flatten(),
            self.fc1
        )

    def apply_mask(self, x):
        batch_size, _, _ = x.shape
        mask = torch.rand(batch_size, self.input_channels, self.sequence_length, device=x.device) > self.mask_ratio
        return x * mask, mask

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_masked, mask = self.apply_mask(x)
        x_hat = self.forward(x_masked)
        loss = self.criterion(x_hat * mask, x * mask)  # Compute loss only on unmasked parts
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_masked, mask = self.apply_mask(x)
        x_hat = self.forward(x_masked)
        loss = self.criterion(x_hat * mask, x * mask)  # Compute loss only on unmasked parts
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

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
