import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(spectrogram, fs=128, hop_length=10):
    spectrogram = spectrogram.numpy()

    freqs = np.linspace(0, fs // 2, spectrogram.shape[0])
    times = np.arange(spectrogram.shape[1]) * hop_length / fs

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, freqs, spectrogram, shading="gouraud")
    plt.colorbar(label="Magnitude")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram")
    plt.show()
