from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch
from torch.utils.data import DataLoader, random_split
from typing import Literal


class EegDataset(Dataset):
    def __init__(
        self,
        dataset,
        sample_duration=2,
        channel_to_keep=None,
        classification_task: Literal["digit", "digit_and_empty", "binary"] = "digit",
        low_cut=14.0,
        high_cut=71.0,
    ):
        dataset = dataset.to_pandas()
        if classification_task == "digit":
            dataset = dataset[dataset["label"] != -1]
        elif classification_task == "binary":
            dataset.loc[dataset["label"] != -1, "label"] = 0
            dataset.loc[dataset["label"] == -1, "label"] = 1
        elif classification_task == "digit_and_empty":
            dataset.loc[dataset["label"] == -1, "label"] = 10
        else:
            assert False, "Invalid classification task"
        features_names = [col for col in dataset.columns if "label" not in col]
        self.electrodes = list(set([f.split("-")[0] for f in features_names]))
        self.nb_channel = len(self.electrodes)
        self.nb_samples = (max([int(f.split("-")[1]) for f in features_names])) + 1
        self.sampling_rate = self.nb_samples / sample_duration
        first_feature_index = dataset.columns.get_loc(features_names[0])
        dataset = dataset.to_numpy()
        self.labels = list(dataset[:, 0].astype(int))
        self.features = dataset[:, first_feature_index:]
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.features = self._keep_channels(self.features, channel_to_keep)
        self.features = self._apply_filters(self.features).copy()
        self.features = self._norm(self.features)
        self.features = torch.tensor(self.features).float()

    def _keep_channels(self, features, channels):
        self.nb_channel = len(self.electrodes)
        features = features.reshape(-1, self.nb_channel, self.nb_samples)
        if channels:
            self.nb_channel = len(channels)
            features = features[:, [channels], :]
        return features

    def _norm(self, features: np.ndarray):
        means = np.mean(features, axis=-1, keepdims=True)
        stds = np.std(features, axis=-1, keepdims=True)

        stds[stds == 0] = 1e-6
        features = (features - means) / stds

        return features

    def _apply_filters(self, features: np.ndarray):
        nyquist = 0.5 * self.sampling_rate

        # Band-pass filter
        low_cut = self.low_cut / nyquist
        high_cut = self.high_cut / nyquist
        b_band, a_band = butter(5, [low_cut, high_cut], btype="band")
        features = filtfilt(b_band, a_band, features, axis=-1)

        # Notch filter for 50Hz
        notch_freq = 50.0
        quality_factor = 30  # Higher Q = narrower bandwidth
        w0 = notch_freq / nyquist
        b_notch, a_notch = iirnotch(w0, quality_factor)
        features = filtfilt(b_notch, a_notch, features, axis=-1)

        return features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class EegRawDataset(EegDataset):
    def __init__(
        self,
        dataset,
        sample_duration=2,
        channel_to_keep=None,
        classification_task: Literal["digit", "digit_and_empty", "binary"] = "binary",
        low_cut=14.0,
        high_cut=71.0,
    ):
        super().__init__(
            dataset=dataset,
            sample_duration=sample_duration,
            channel_to_keep=channel_to_keep,
            classification_task=classification_task,
            low_cut=low_cut,
            high_cut=high_cut,
        )

    def collate_to_2d_fn(self, batch):
        x, y = zip(*batch)
        return torch.cat(x).view(-1, 1, self.nb_channel, self.nb_samples)[
            :, :, :, :
        ], torch.tensor(y).long()


class EegSpecDataset(EegDataset):
    def __init__(
        self,
        dataset,
        sample_duration=2,
        channel_to_keep=None,
        classification_task: Literal["digit", "digit_and_empty", "binary"] = "binary",
        low_cut=14.0,
        high_cut=71.0,
        hop_length=64,
    ):
        super().__init__(
            dataset=dataset,
            sample_duration=sample_duration,
            channel_to_keep=channel_to_keep,
            classification_task=classification_task,
            low_cut=low_cut,
            high_cut=high_cut,
        )
        self.hop_length = hop_length
        self.features = self._spectograms(self.features)
        self.features = self._min_max_normalize_spectrograms(self.features)
        self.features = self.features.unflatten(2,(3,-1)) # RGB Like images for each channel

    def _spectograms(self, features, n_fft=256):
        window = torch.hann_window(n_fft)
        batch_size, nb_channel, signal_length = features.shape
        features_reshaped = features.view(-1, signal_length)

        stft_result = torch.stft(
            features_reshaped,
            n_fft=n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            normalized=True,
        )
        spectrograms = torch.abs(stft_result)

        freq_bins, time_frames = spectrograms.shape[1], spectrograms.shape[2]
        spectrograms = spectrograms.view(batch_size, nb_channel, freq_bins, time_frames)
        return spectrograms

    def _min_max_normalize_spectrograms(self, spectrograms: torch.Tensor):
        min_vals = spectrograms.amin(dim=(-2, -1), keepdim=True)
        max_vals = spectrograms.amax(dim=(-2, -1), keepdim=True)
        eps = 1e-6
        denominator = (max_vals - min_vals).clamp(min=eps)
        specs_norm = (spectrograms - min_vals) / denominator
        return specs_norm

    def _standard_normalize_spectrograms(self, spectrograms: torch.Tensor):
        mean = spectrograms.mean(dim=(-2, -1), keepdim=True)
        std = spectrograms.std(dim=(-2, -1), keepdim=True)
        eps = 1e-6
        std = std.clamp(min=eps)
        specs_norm = (spectrograms - mean) / std
        return specs_norm


def create_dataloaders(
    dataset, batch_size=128, train_split=0.8, seed=42, two_dimension=False
):
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # Use random_split to create train and validation datasets
    # train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    # val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # Create DataLoaders for the train and validation datasets
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_to_2d_fn if two_dimension else None,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_to_2d_fn if two_dimension else None,
    )

    return train_dataloader, val_dataloader
