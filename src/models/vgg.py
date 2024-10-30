import torch
import torch.nn as nn
import pytorch_lightning as pl

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, pool_kernel=2):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # Update in_channels for the next conv layer
        layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=2))  # Downsampling
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VGGish(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(VGGish, self).__init__()
        
        # Define VGG-like convolutional blocks using ConvBlock
        self.features = nn.Sequential(
            ConvBlock(5, 64, num_convs=2),    # Block 1
            ConvBlock(64, 128, num_convs=2),  # Block 2
            ConvBlock(128, 256, num_convs=3), # Block 3
            ConvBlock(256, 512, num_convs=3), # Block 4
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 1, 4096),  # Adjusted input size based on downsampling
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),  # Output for MNIST (10 classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=2,
            verbose=True,
            min_lr=0.000001,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
