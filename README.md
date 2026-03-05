# EEG 噪声类型分类项目

基于深度学习的单通道 EEG 信号噪声类型分类系统。使用 EEGdenoiseNet 数据集，将 EEG 信号按污染类型分为三类：**干净 EEG**、**眼电（EOG）污染**、**肌电（EMG）污染**，并对比多种网络结构（CNN、ResNet1D、CNN-LSTM、CNN-Attention）的分类性能。

---

## 环境配置

### 依赖版本

| 依赖 | 版本 |
|------|------|
| Python | 3.6+ |
| PyTorch | 1.9.0+ |
| MNE | 0.22.1 |
| NumPy / SciPy | - |
| scikit-learn | - |
| matplotlib / seaborn | - |
| pandas | - |

### 安装

```bash
# 建议使用 conda 或 venv 创建虚拟环境
pip install torch==1.9.0
pip install MNE==0.22.1
pip install numpy scipy scikit-learn matplotlib seaborn pandas
```

---

## 数据集

本项目使用 **EEGdenoiseNet** 基准数据集，适用于训练与评估基于深度学习的 EEG 去噪与相关任务。

- **论文**：[Journal of Neural Engineering](https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8)
- **数据下载**：[G-node](https://gin.g-node.org/NCClab/EEGdenoiseNet)（512 Hz 的 EEG/EMG epoch 数据）
- **相关工具箱**：[Single-Channel-EEG-Denoise](https://github.com/ncclabsustech/Single-Channel-EEG-Denoise)

数据包含：
- `EEG_all_epochs.npy`：干净 EEG 片段
- `EOG_all_epochs.npy`：眼电噪声片段
- `EMG_all_epochs.npy`：肌电噪声片段

请将上述 `.npy` 文件放在 `data/` 目录下，或在脚本中修改为你的数据路径。

---

## 项目结构

```
code_main/
├── data/                          # 数据与数据生成脚本
│   ├── data_input.py              # 数据加载、EEG/噪声混合、标准化
│   ├── generate_data.py           # 生成去噪任务用数据
│   ├── generate_classification_data.py  # 生成 3 类分类数据
│   └── classify_data.py           # 分类数据准备（备用流程）
├── code/                          # 模型与训练脚本
│   ├── classification_network.py  # CNN 分类器 (EEGClassifier)
│   ├── resnet1d_classifier.py     # ResNet1D 分类器
│   ├── cnn_lstm_classifier.py     # CNN-LSTM 分类器
│   ├── attention_classifier.py    # CNN-Attention 分类器
│   ├── evaluation_utils.py        # 评估与可视化工具
│   ├── train_classifier.py        # 训练 CNN
│   ├── train_resnet1d.py          # 训练 ResNet1D
│   ├── train_cnn_lstm.py          # 训练 CNN-LSTM
│   └── train_attention.py         # 训练 CNN-Attention
├── results/                       # 各模型结果与汇总
│   ├── cnn_model/
│   ├── resnet1d_model/
│   ├── cnn_lstm_model/
│   └── attention_model/
└── README.md
```

---

## 使用流程

### 1. 准备数据

确保已下载 EEGdenoiseNet 的 `EEG_all_epochs.npy`、`EOG_all_epochs.npy`、`EMG_all_epochs.npy`，并放在 `data/` 或修改脚本中的路径。

**生成分类数据集（3 类：Clean / EOG / EMG）：**

```bash
cd data
python generate_classification_data.py
```

会生成 `classification_signals.npy` 与 `classification_labels.npy`。若使用 `classify_data.py` 或自定义路径，请在对应脚本中修改输入/输出路径。

### 2. 训练模型

在 `code/` 目录下运行对应训练脚本（需能访问到数据与标签路径，必要时在脚本内修改路径）：

```bash
cd code

# CNN 分类器
python train_classifier.py

# ResNet1D
python train_resnet1d.py

# CNN-LSTM
python train_cnn_lstm.py

# CNN-Attention
python train_attention.py
```

训练采用 **5 折分层交叉验证**，并使用类别加权采样缓解类别不平衡。每个模型会在 `results/` 下对应子目录中保存指标、混淆矩阵、ROC 曲线及 t-SNE 可视化等。

### 3. 查看结果

各模型的结果汇总在：

- `results/cnn_model/aggregated/CNN_Model_summary.txt`
- `results/resnet1d_model/aggregated/ResNet1D_Model_summary.txt`
- `results/cnn_lstm_model/aggregated/CNN_LSTM_Model_summary.txt`
- `results/attention_model/aggregated/CNN_Attention_Model_summary.txt`

其中包含整体准确率、各类别精确率/召回率、混淆矩阵及分类报告等。

---

## 模型简介

| 模型 | 说明 |
|------|------|
| **CNN (EEGClassifier)** | 4 层 1D 卷积 + BN + 池化 + 全连接，输入长度 512，输出 3 类 |
| **ResNet1D** | 1D ResNet 结构，适合较长序列特征提取 |
| **CNN-LSTM** | 1D CNN 提取特征后经双向 LSTM 与全连接层分类 |
| **CNN-Attention** | CNN 特征 + 自注意力机制，突出关键时间/通道信息 |

分类目标为三分类：**0-干净 EEG**、**1-EOG 污染**、**2-EMG 污染**。

---

## 数据与标签说明

- **类别 0**：干净 EEG  
- **类别 1**：以 EOG 污染为主的 EEG（如纯 EEG+EOG、或 EOG 占主导的混合）  
- **类别 2**：以 EMG 污染为主的 EEG（如纯 EEG+EMG、或 EMG 占主导的混合）  

混合时通过设定信噪比（如 -7 dB～2 dB）控制噪声强度，数据生成逻辑见 `data/generate_classification_data.py` 与 `data/data_input.py`。

---

## 参考文献与数据来源

- EEGdenoiseNet 数据集论文：*EEGdenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising*，Journal of Neural Engineering，2022  
- 数据集与代码：[EEGdenoiseNet (G-node)](https://gin.g-node.org/NCClab/EEGdenoiseNet)，[Single-Channel-EEG-Denoise](https://github.com/ncclabsustech/Single-Channel-EEG-Denoise)

---

## 说明

- 若数据路径与上述不同，请在 `data/generate_data.py`、`data/generate_classification_data.py` 以及各 `code/train_*.py` 中修改为本地路径。  
- 本项目仅供学习。
