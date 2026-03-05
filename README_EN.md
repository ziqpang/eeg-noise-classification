# EEG Noise Type Classification

A deep learning–based system for classifying single-channel EEG signals by noise type. Using the **EEGdenoiseNet** dataset, it classifies EEG into three categories: **clean EEG**, **EOG (electrooculography) contamination**, and **EMG (electromyography) contamination**, and compares the performance of several network architectures: CNN, ResNet1D, CNN-LSTM, and CNN-Attention.

---

## Environment Setup

### Dependencies

| Dependency            | Version |
| --------------------- | ------- |
| Python                | 3.6+    |
| PyTorch               | 1.9.0+  |
| MNE                   | 0.22.1  |
| NumPy / SciPy         | -       |
| scikit-learn         | -       |
| matplotlib / seaborn  | -       |
| pandas                | -       |

### Installation

```bash
# Recommended: use conda or venv for a virtual environment
pip install torch==1.9.0
pip install MNE==0.22.1
pip install numpy scipy scikit-learn matplotlib seaborn pandas
```

---

## Dataset

This project uses the **EEGdenoiseNet** benchmark dataset for training and evaluating deep learning–based EEG denoising and related tasks.

- **Paper**: [Journal of Neural Engineering](https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8)
- **Data**: [G-node](https://gin.g-node.org/NCClab/EEGdenoiseNet) (EEG/EMG epochs at 512 Hz)
- **Toolbox**: [Single-Channel-EEG-Denoise](https://github.com/ncclabsustech/Single-Channel-EEG-Denoise)

Required files:

- `EEG_all_epochs.npy` — clean EEG epochs
- `EOG_all_epochs.npy` — EOG (eye) noise epochs
- `EMG_all_epochs.npy` — EMG (muscle) noise epochs

Place these `.npy` files in the `data/` directory, or update the paths in the scripts.

---

## Project Structure

```
code_main/
├── data/                                  # Data and data-generation scripts
│   ├── data_input.py                      # Load data, mix EEG/noise, standardize
│   ├── generate_data.py                   # Generate denoising-task data
│   ├── generate_classification_data.py    # Generate 3-class classification data
│   └── classify_data.py                   # Alternative classification data pipeline
├── code/                                  # Models and training scripts
│   ├── classification_network.py          # CNN classifier (EEGClassifier)
│   ├── resnet1d_classifier.py             # ResNet1D classifier
│   ├── cnn_lstm_classifier.py             # CNN-LSTM classifier
│   ├── attention_classifier.py            # CNN-Attention classifier
│   ├── evaluation_utils.py                # Evaluation and visualization
│   ├── train_classifier.py                # Train CNN
│   ├── train_resnet1d.py                  # Train ResNet1D
│   ├── train_cnn_lstm.py                  # Train CNN-LSTM
│   └── train_attention.py                 # Train CNN-Attention
├── results/                               # Model results and summaries
│   ├── cnn_model/
│   ├── resnet1d_model/
│   ├── cnn_lstm_model/
│   └── attention_model/
└── README.md
```

---

## Usage

### 1. Prepare data

Download EEGdenoiseNet’s `EEG_all_epochs.npy`, `EOG_all_epochs.npy`, and `EMG_all_epochs.npy`, and place them in `data/` or adjust paths in the scripts.

**Generate the 3-class classification dataset (Clean / EOG / EMG):**

```bash
cd data
python generate_classification_data.py
```

This produces `classification_signals.npy` and `classification_labels.npy`. If you use `classify_data.py` or custom paths, update the input/output paths in the corresponding scripts.

### 2. Train models

From the `code/` directory, run the desired training script (ensure data and label paths are correct; edit paths in the script if needed):

```bash
cd code

# CNN classifier
python train_classifier.py

# ResNet1D
python train_resnet1d.py

# CNN-LSTM
python train_cnn_lstm.py

# CNN-Attention
python train_attention.py
```

Training uses **5-fold stratified cross-validation** and class-weighted sampling to handle class imbalance. Each model writes metrics, confusion matrices, ROC curves, and t-SNE plots to its subfolder under `results/`.

### 3. View results

Summary files for each model:

- `results/cnn_model/aggregated/CNN_Model_summary.txt`
- `results/resnet1d_model/aggregated/ResNet1D_Model_summary.txt`
- `results/cnn_lstm_model/aggregated/CNN_LSTM_Model_summary.txt`
- `results/attention_model/aggregated/CNN_Attention_Model_summary.txt`

They include overall accuracy, per-class precision/recall, confusion matrices, and classification reports.

---

## Models

| Model                   | Description |
| ----------------------- | ----------- |
| **CNN (EEGClassifier)** | Four 1D conv blocks + BN + pooling + FC; input length 512, 3-class output |
| **ResNet1D**            | 1D ResNet for longer-sequence feature extraction |
| **CNN-LSTM**            | 1D CNN features fed into bidirectional LSTM and FC for classification |
| **CNN-Attention**       | CNN features + self-attention to emphasize key time/channel information |

Labels: **0** — clean EEG, **1** — EOG contamination, **2** — EMG contamination.

---

## Data and labels

- **Class 0**: Clean EEG  
- **Class 1**: EOG-dominated (e.g. EEG+EOG or EOG-dominant mixture)  
- **Class 2**: EMG-dominated (e.g. EEG+EMG or EMG-dominant mixture)

Noise level is controlled by SNR (e.g. −7 dB to 2 dB). See `data/generate_classification_data.py` and `data/data_input.py` for the generation logic.

---

## References

- EEGdenoiseNet: *EEGdenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising*, Journal of Neural Engineering, 2022  
- Dataset: [EEGdenoiseNet (G-node)](https://gin.g-node.org/NCClab/EEGdenoiseNet)

## Notes

- If your data paths differ, update them in `data/generate_data.py`, `data/generate_classification_data.py`, and the `code/train_*.py` scripts.  
- This project is for educational use.
