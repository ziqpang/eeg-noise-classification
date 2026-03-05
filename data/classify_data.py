import numpy as np
from data_input import get_rms, random_signal, data_prepare
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

def generate_mixed_signals(EEG_all, EOG_all, EMG_all, combin_num, train_num, test_num):
    """
    生成混合信号数据集
    Args:
        EEG_all: 原始EEG信号
        EOG_all: 眼电信号
        EMG_all: 肌电信号
        combin_num: 每个样本的混合次数
        train_num: 训练集样本数
        test_num: 测试集样本数
    Returns:
        五种类型的信号：clean_eeg, eog_eeg, emg_eeg, emg_eog_eeg, eog_emg_eeg
    """
    # 计算每个类别的样本数
    samples_per_class = min(train_num, EEG_all.shape[0] // 5)
    
    # 生成干净EEG信号
    clean_eeg = EEG_all[0:samples_per_class, :]
    
    # 生成EEG+EOG混合信号
    eog_eeg, _, _, _, _ = data_prepare(EEG_all[samples_per_class:2*samples_per_class], 
                                      EOG_all[0:samples_per_class], 
                                      combin_num, samples_per_class, 0)
    
    # 生成EEG+EMG混合信号
    emg_eeg, _, _, _, _ = data_prepare(EEG_all[2*samples_per_class:3*samples_per_class], 
                                      EMG_all[0:samples_per_class], 
                                      combin_num, samples_per_class, 0)
    
    # 生成EEG+EMG+小EOG混合信号
    eog_small = EOG_all[samples_per_class:2*samples_per_class] * 0.3  # 降低EOG幅度
    emg_eog_eeg, _, _, _, _ = data_prepare(EEG_all[3*samples_per_class:4*samples_per_class], 
                                          eog_small, 
                                          combin_num, samples_per_class, 0)
    emg_eog_eeg, _, _, _, _ = data_prepare(emg_eog_eeg, 
                                          EMG_all[samples_per_class:2*samples_per_class], 
                                          combin_num, samples_per_class, 0)
    
    # 生成EEG+小EMG+EOG混合信号
    emg_small = EMG_all[2*samples_per_class:3*samples_per_class] * 0.3  # 降低EMG幅度
    eog_emg_eeg, _, _, _, _ = data_prepare(EEG_all[4*samples_per_class:5*samples_per_class], 
                                          emg_small, 
                                          combin_num, samples_per_class, 0)
    eog_emg_eeg, _, _, _, _ = data_prepare(eog_emg_eeg, 
                                          EOG_all[2*samples_per_class:3*samples_per_class], 
                                          combin_num, samples_per_class, 0)
    
    return clean_eeg, eog_eeg, emg_eeg, emg_eog_eeg, eog_emg_eeg

def prepare_classification_data(EEG_all, EOG_all, EMG_all, combin_num, train_num, test_num):
    """
    准备分类数据集
    Args:
        EEG_all: 原始EEG信号
        EOG_all: 眼电信号
        EMG_all: 肌电信号
        combin_num: 每个样本的混合次数
        train_num: 训练集样本数
        test_num: 测试集样本数
    Returns:
        X: 特征数据
        y: 标签数据 (0: 无噪声, 1: 有眼电噪声, 2: 有肌电噪声)
    """
    # 生成所有类型的信号
    clean_eeg, eog_eeg, emg_eeg, emg_eog_eeg, eog_emg_eeg = generate_mixed_signals(
        EEG_all, EOG_all, EMG_all, combin_num, train_num, test_num)
    
    # 计算每个类别的最小样本数
    min_samples = min(clean_eeg.shape[0], eog_eeg.shape[0], emg_eeg.shape[0], 
                     emg_eog_eeg.shape[0], eog_emg_eeg.shape[0])
    
    # 从每个类别中随机选择相同数量的样本
    clean_indices = np.random.choice(clean_eeg.shape[0], min_samples, replace=False)
    eog_indices = np.random.choice(eog_eeg.shape[0], min_samples, replace=False)
    emg_indices = np.random.choice(emg_eeg.shape[0], min_samples, replace=False)
    emg_eog_indices = np.random.choice(emg_eog_eeg.shape[0], min_samples, replace=False)
    eog_emg_indices = np.random.choice(eog_emg_eeg.shape[0], min_samples, replace=False)
    
    # 选择平衡的样本
    clean_eeg = clean_eeg[clean_indices]
    eog_eeg = eog_eeg[eog_indices]
    emg_eeg = emg_eeg[emg_indices]
    emg_eog_eeg = emg_eog_eeg[emg_eog_indices]
    eog_emg_eeg = eog_emg_eeg[eog_emg_indices]
    
    # 创建标签
    # 0: 无噪声, 1: 有眼电噪声, 2: 有肌电噪声
    clean_labels = np.zeros(min_samples)
    eog_labels = np.ones(min_samples)  # 标记为有眼电噪声
    emg_labels = np.ones(min_samples) * 2  # 标记为有肌电噪声
    emg_eog_labels = np.ones(min_samples)  # 标记为有眼电噪声
    eog_emg_labels = np.ones(min_samples) * 2  # 标记为有肌电噪声
    
    # 合并所有数据
    X = np.vstack((clean_eeg, eog_eeg, emg_eeg, emg_eog_eeg, eog_emg_eeg))
    y = np.concatenate((clean_labels, eog_labels, emg_labels, emg_eog_labels, eog_emg_labels))
    
    # 打乱数据
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    
    # 打印数据集信息
    print(f"每个类别的样本数: {min_samples}")
    print(f"总样本数: {X.shape[0]}")
    print(f"类别分布: {np.bincount(y.astype(int))}")
    
    return X, y

def plot_confusion_matrix(y_true, y_pred, fold_idx):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold_idx}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'output_images/confusion_matrix_fold_{fold_idx}.png')
    plt.close()

def plot_roc_curve(y_true, y_score, fold_idx):
    """
    绘制ROC曲线
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_idx}')
    plt.legend(loc="lower right")
    plt.savefig(f'output_images/roc_curve_fold_{fold_idx}.png')
    plt.close()

def main():
    """
    主函数：加载数据并准备10折交叉验证数据集
    """
    # 加载原始数据
    EEG_all = np.load('/root/autodl-tmp/DeepSeparator-main/data/EEG_all_epochs.npy')
    EOG_all = np.load('/root/autodl-tmp/DeepSeparator-main/data/EOG_all_epochs.npy')
    EMG_all = np.load('/root/autodl-tmp/DeepSeparator-main/data/EMG_all_epochs.npy')
    
    # 参数设置
    combin_num = 10  # 每个样本的混合次数
    train_num = 3000  # 训练集样本数
    test_num = 400   # 测试集样本数
    
    # 准备数据
    X, y = prepare_classification_data(EEG_all, EOG_all, EMG_all, combin_num, train_num, test_num)
    
    # 创建输出目录
    os.makedirs('output_images', exist_ok=True)
    os.makedirs('data/classification', exist_ok=True)
    
    # 10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 保存数据
        np.save(f'data/classification/X_train_fold_{fold_idx}.npy', X_train)
        np.save(f'data/classification/y_train_fold_{fold_idx}.npy', y_train)
        np.save(f'data/classification/X_test_fold_{fold_idx}.npy', X_test)
        np.save(f'data/classification/y_test_fold_{fold_idx}.npy', y_test)
        
        # 打印每折的类别分布
        print(f"\n第{fold_idx}折类别分布:")
        print(f"训练集: {np.bincount(y_train.astype(int))}")
        print(f"测试集: {np.bincount(y_test.astype(int))}")

if __name__ == "__main__":
    main()