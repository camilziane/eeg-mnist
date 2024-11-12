# EEG-MNIST: Reproducing Brain Signal Models

Welcome to **EEG-MNIST**, a project aiming to reproduce the results of top-performing models on brain signal datasets as showcased in the [Hugging Face MindBigData Leaderboard](https://huggingface.co/spaces/DavidVivancos/MindBigData-Leaderboard).

We are currently focusing on implementing convolutional neural networks (CNNs) as a foundational step before moving on to more advanced approaches. This project explores brain EEG models and datasets provided by **David Vivancos**, particularly the: **MindBigData 2022: A Large Dataset of Brain Signals**

---

## 🧠 Project Overview

This project is built to validate and test EEG models from peer-reviewed papers. For now, the implementations include:

1. **CNN-based model** from [Sensors (2023)](https://doi.org/10.3390/s23239351)
2. **SPENet** from [Neurocomputing (2024)](https://doi.org/10.1016/j.neucom.2024.127654)

While faithfully replicating the papers, our results currently fall short of their claimed performance (13–15% accuracy compared to >80%). This discrepancy might stem from:

- Potential misunderstandings of the papers.
- Preprocessing differences.

Our goal is to identify these gaps and refine the implementations to match or exceed the reported results.

---

## 📦 Getting Started

Follow these steps to set up and run the project:

### 1. Clone and Sync the Project

```bash
git clone https://github.com/camilziane/eeg-mnist.git
cd eeg-mnist
uv sync
```

### 2. Explore and Test Implementations

Open the `main.ipynb` notebook to test the models implemented so far:

- CNN-based model
- SPENet

Run the cells to load the dataset, preprocess the data, and evaluate the models.

---

## 📄 Datasets

### MindBigData 2022 & 2023

The EEG-MNIST dataset was created by **David Vivancos** as part of the **MindBigData** initiative. These datasets are among the largest and most detailed publicly available collections of brain signals.

More details:

- [MindBigData 2022](https://huggingface.co/datasets/DavidVivancos/MindBigData2022)

---

## 🚠 Future Directions

This project is an ongoing exploration of EEG-based modeling. Upcoming steps include:

- Refining CNN implementations for higher accuracy.
- Incorporating advanced techniques such as transformer-based architectures.
- Scaling to **state-of-the-art (SOTA)** models on the leaderboard.

---

## 🤖 Contributing

Feel free to fork this repository and submit pull requests with improvements, bug fixes, or additional model implementations. Let's build something great together!

---

## ⚠️ Disclaimer

This project is for research and learning purposes. The results might not yet align with the original papers due to preprocessing, implementation differences, or other factors. Contributions and suggestions to address these gaps are highly welcome.

---

## 🌟 Acknowledgments

Special thanks to:

- **David Vivancos** for creating and sharing the MindBigData datasets.
- Authors of the referenced papers for their innovative research in EEG modeling.

