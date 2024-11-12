# reference https://doi.org/ 10.3390/s23239351
import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5, 1),
        stride=1,
        padding=(2, 0),
        dropout_rate=0.2,
        pool_kernel=(5, 1),
        pool_stride=(5, 1),
        residual=False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if pool_kernel is None:
            self.pool = nn.Identity()
        else:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class CNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_channels=1,
        sequence_length=256,
        sequence_nb_channel=14,
        num_classes=10,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_nb_channel = sequence_nb_channel
        self.sequence_length = sequence_length
        kernel_size = (5, 1)
        padding = (2, 0)
        pool_kernel = (1, 1)
        pool_stride = (1, 1)

        self.block1 = ConvBlock2d(
            input_channels,
            14,
            kernel_size=kernel_size,
            padding=padding,
            dropout_rate=dropout_rate,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )

        self.block2 = ConvBlock2d(
            14,
            10,
            kernel_size=kernel_size,
            padding=padding,
            dropout_rate=dropout_rate,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )

        self.block3 = ConvBlock2d(
            10,
            10,
            kernel_size=kernel_size,
            padding=padding,
            dropout_rate=dropout_rate,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )

        self.fc_input_size = self._calculate_fc_input_size()

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 3500),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3500, 2500),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Linear(2500, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def _calculate_fc_input_size(self):
        x = torch.randn(
            1, self.input_channels, self.sequence_nb_channel, self.sequence_length
        )

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.mean(dim=1)
        x = x.flatten(1)
        return x.shape[1]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.mean(dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        logits = self.classifier(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, weight_decay=5e-5)

        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=0.000001,
        )

        return [optimizer], [
            {
                "scheduler": plateau_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        ]
