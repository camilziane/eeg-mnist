# reference https://doi.org/10.1016/j.neucom.2024.127654
import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=5, padding=2, stride=1, dropout=0.5
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SpeNet(pl.LightningModule):
    def __init__(self, num_classes=11, num_channels=64, dropout=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.conv1 = ConvBlock(3, 256, dropout=dropout)
        self.conv2 = ConvBlock(256, 256, dropout=dropout)
        self.conv3 = ConvBlock(256, 128, dropout=dropout)
        self.conv4 = ConvBlock(128, 64, dropout=dropout)
        self.maxpool = nn.MaxPool2d(4)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, num_classes),
            nn.Dropout(dropout),
            nn.Softmax(dim=1),
        )

        self.fc2 = nn.Linear(num_channels * num_classes, num_classes)
        # self.criterion = nn.CrossEntropyLoss()

    def set_class_weights(self, train_dataset, alpha=1):
        all_labels = []
        for _, y in train_dataset:
            all_labels.append(y)

        labels = torch.tensor(all_labels)
        label_counts = torch.bincount(labels)
        total_samples = len(labels)

        self.class_weights = (
            total_samples / (self.num_classes * label_counts)
        ) ** alpha
        self.class_weights = self.class_weights.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        batch_size, num_channels, depth, height, width = x.shape

        x = x.reshape(batch_size * num_channels, depth, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x)
        x = x.view(batch_size, -1)
        final_output = self.fc2(x)
        return final_output

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
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4, weight_decay=1e-6)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=500, verbose=True, min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
