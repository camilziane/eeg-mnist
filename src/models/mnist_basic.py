import torch
import torch.nn as nn
import pytorch_lightning as pl

class MNISTClassifier(pl.LightningModule):
    def __init__(self, latent_dim=64, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features), features

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
