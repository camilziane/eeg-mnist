import torch
import torch.nn as nn
import pytorch_lightning as pl

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.mish = nn.Mish()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.mish(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return self.mish(out)

class ResNetClassifier(pl.LightningModule):
    def __init__(self, input_channels=5, sequence_length=256, num_classes=10, dropout_rate=0.5):  # Higher dropout
        super(ResNetClassifier, self).__init__()
        
        # Initial layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.Mish(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Reduced Residual Blocks
        self.layer1 = ResidualBlock(16, 32, downsample=True)
        self.layer2 = ResidualBlock(32, 64, downsample=True)
        
        # Calculate size before FC layers
        reduced_length = sequence_length // 4
        self.fc_input_size = 1024
        
        # Fully connected layers with residual connection
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 256),  # Reduced size
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),  # Reduced size
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.classifier = nn.Linear(256, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Residual connection in fully connected layers
        x = self.fc1(x)
        x = x + self.fc2(x)
        logits = self.classifier(x)
        
        return logits

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01, weight_decay=1e-3)
        
        # Cosine Annealing Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        return [optimizer], [scheduler]
