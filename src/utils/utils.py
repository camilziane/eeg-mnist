from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())



def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final losses
    print(f"Final training loss: {train_losses[-1]}")
    print(f"Final validation loss: {val_losses[-1]}")


def plot_spectrogram(spectrogram, batch_idx, channel_idx, fs=128, hop_length=10, freq_limit=(0.5, 30)):
    """
    Plots the spectrogram for a given batch and channel.

    Parameters:
    - spectrogram: The spectrogram data (numpy array).
    - batch_idx: Index of the batch/sample to plot.
    - channel_idx: Index of the channel to plot.
    - fs: Sampling rate (default is 128).
    - hop_length: Hop length for the spectrogram (default is 1).
    - freq_limit: Tuple indicating the frequency range to display (default is (0.5, 5) Hz).
    """
    # Extract the spectrogram for the selected sample and channel
    sample_spectrogram = spectrogram[batch_idx, channel_idx].numpy()  # Shape: (freq_bins, time_steps)

    # Generate frequency and time axes to match the shape of `sample_spectrogram`
    freqs = np.linspace(0, fs // 2, sample_spectrogram.shape[0])  # Frequency axis
    times = np.arange(sample_spectrogram.shape[1]) * hop_length / fs  # Time axis

    # Filter frequencies to show only from 1 Hz
    freq_mask = (freqs > freq_limit[0]) & (freqs <= freq_limit[1])
    filtered_spectrogram = sample_spectrogram[freq_mask, :]
    filtered_freqs = freqs[freq_mask]

    # Plot the spectrogram for visualization
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, filtered_freqs, filtered_spectrogram, shading='gouraud')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title(f'Spectrogram - Sample {batch_idx}, Channel {channel_idx} ({freq_limit[0]} Hz and above)')
    plt.ylim(freq_limit)  # Optionally limit to the specified frequency range
    plt.show()


def create_dataloaders(dataset, batch_size=128, train_split=0.8, seed=42):
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # Use random_split to create train and validation datasets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(seed))

    # Create DataLoaders for the train and validation datasets
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, val_dataloader
