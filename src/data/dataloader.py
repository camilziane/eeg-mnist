
from torch.utils.data import Dataset
import numpy as np
import torch
import random

class EEG_Dataset(Dataset):
    def __init__(self, dataset):
        data_df = dataset.to_pandas()
        features_names = list(data_df.columns)[1:]
        self.electrodes = list(set([f.split("-")[0] for f in features_names]))
        self.nb_channel = len(self.electrodes)
        self.nb_samples = (max([int(f.split("-")[1]) for f in features_names]))+1
        self.datas = self._norm(data_df)
        self.features = self.datas[:,1:]
        self.labels = list(self.datas[:,0].astype(int))

    def _norm(self, data_df):
        electrodes_min = {}
        electrodes_max = {}
        for electrode in self.electrodes :
            electrodes_min[electrode] = data_df[[f'{electrode}-{i}'for i in range(self.nb_samples)]].to_numpy().reshape(-1,).max()
            electrodes_max[electrode] = data_df[[f'{electrode}-{i}'for i in range(self.nb_samples)]].to_numpy().reshape(-1,).min()

        for electrode in self.electrodes:
            data_df[[f'{electrode}-{i}'for i in range(self.nb_samples)]] = (data_df[[f'{electrode}-{i}'for i in range(self.nb_samples)]] - electrodes_min[electrode])/(electrodes_max[electrode] -electrodes_min[electrode])
        return data_df.to_numpy()
      
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]).view(self.nb_channel,-1).float(), self.labels[idx]

class EEG_Spectogram_Dataset(Dataset):
    def __init__(self, dataset):
        data_df = dataset.to_pandas()
        features_names = list(data_df.columns)[1:]
        self.electrodes = list(set([f.split("-")[0] for f in features_names]))
        self.nb_channel = len(self.electrodes)
        self.nb_samples = (max([int(f.split("-")[1]) for f in features_names])) + 1
        datas = data_df.to_numpy()
        self.features = datas[:, 1:]
        self.labels = list(datas[:, 0].astype(int))
        self.spectrograms = self._norm(self._spectogram(features=self.features)).float() 
        del datas

    def _spectogram(self, features, n_fft=256, hop_length=10, fs=128, lowcut=0.5):
        features = torch.tensor(
            features.reshape(-1, features.shape[-1] // self.nb_channel)
        )
        window = torch.hann_window(n_fft)
        stft_result = torch.stft(
            features,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        spectrogram = torch.abs(stft_result)
        spectrogram = spectrogram.reshape(
            -1, self.nb_channel, spectrogram.shape[-2], spectrogram.shape[-1]
        )

        freqs = np.linspace(0, fs // 2, spectrogram.shape[2])
        start_freq = np.argwhere(freqs > lowcut)[0].item()
        spectrogram = spectrogram[:, :, start_freq:, :]
        return spectrogram

    def _norm(self, spectrogram):
        spectrogram_min = torch.amin(spectrogram, dim=(0, 2, 3), keepdim=True)
        spectrogram_max = torch.amax(spectrogram, dim=(0, 2, 3), keepdim=True)
        return (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]

class EncodedDataset:
    def __init__(self, dataloader, encoder):
        super().__init__()
        self.encoded_datas, self.labels = self._encode_dataset(dataloader, encoder)

    def _encode_dataset(self, dataloader, encoder):
        encoded_datas = []
        labels = []
        for batch in dataloader:
            encoded_data = encoder(batch[0]).detach()
            encoded_datas.append(encoded_data)
            labels.append(batch[1])
        encoded_datas = torch.cat(encoded_datas)
        labels = torch.cat(labels)
        return encoded_datas, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded_datas[idx], self.labels[idx].item()

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_data, mnist_data):
        self.paired_data = self._pair_data(eeg_data, mnist_data)

    def _pair_data(self, eeg_data, mnist_data):
        eeg_by_label = {i: [] for i in range(10)}
        mnist_by_label = {i: [] for i in range(10)}

        # Group EEG data by label
        for eeg, label in eeg_data: 
            if 0 <= label < 10:
                eeg_by_label[label].append(eeg)

        # Group MNIST data by label
        for img, label in mnist_data:
            mnist_by_label[label].append(img)

        # Pair EEG and MNIST data
        paired_data = []
        for label in eeg_by_label.keys():
            eeg_samples = eeg_by_label[label]
            mnist_samples = mnist_by_label[label]

            # Use the maximum number of samples available
            n_samples = max(len(eeg_samples), len(mnist_samples))

            # Replicate samples if necessary
            if len(eeg_samples) < n_samples:
                eeg_samples = self._replicate_samples(eeg_samples, n_samples)
            if len(mnist_samples) < n_samples:
                mnist_samples = self._replicate_samples(mnist_samples, n_samples)

            for eeg, mnist in zip(eeg_samples, mnist_samples):
                paired_data.append((eeg, mnist, label))

        return paired_data

    def _replicate_samples(self, samples, target_size):
        """Replicate samples randomly to reach the target size."""
        while len(samples) < target_size:
            samples.append(random.choice(samples))
        return samples

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        return self.paired_data[idx]



