import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNNClassifier(pl.LightningModule):
    def __init__(self, input_channels=5, sequence_length=256, num_classes=10, dropout_rate=0.2):
        super(CNNClassifier, self).__init__()
        
        # Layer 1: Conv + LeakyReLU
        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=5, stride=1, padding=2)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.1)
        self.norm1 = nn.BatchNorm1d(input_channels)
        
        # Layer 3: Conv + LeakyReLU
        self.conv2 = nn.Conv1d(input_channels, 3, kernel_size=5, stride=1, padding=2)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.1)
        self.norm2 = nn.BatchNorm1d(3)
        
        # Layer 5: Conv + LeakyReLU
        self.conv3 = nn.Conv1d(3, 3, kernel_size=5, stride=1, padding=2)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.1)
        self.norm3 = nn.BatchNorm1d(3)
        
        # Calculate size before FC layers
        self.fc_input_size = 3 * sequence_length  # 10 channels * sequence_length
        
        # Layer 7: Full-connected
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, self.fc_input_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout_rate)
        )
        
        # Layer 8: Full-connected
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_input_size//4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.classifier = nn.Linear(self.fc_input_size//4, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.leaky1(x)
        x = self.norm1(x)
        
        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.norm2(x)
        
        x = self.conv3(x)
        x = self.leaky3(x)
        x = self.norm3(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Classification head
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2, verbose=True, min_lr=0.000001,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
