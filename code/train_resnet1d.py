import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from resnet1d_classifier import EEGResNet
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score
from sklearn.manifold import TSNE

# Standardize training and test data per fold
def standardize_data(train_data, test_data):
    scaler = StandardScaler()
    orig_train_shape = train_data.shape
    orig_test_shape = test_data.shape
    train_flat = train_data.reshape(-1, train_data.shape[-1])
    test_flat = test_data.reshape(-1, test_data.shape[-1])
    scaler.fit(train_flat)
    train_scaled = scaler.transform(train_flat)
    test_scaled = scaler.transform(test_flat)
    return train_scaled.reshape(orig_train_shape), test_scaled.reshape(orig_test_shape)

# Compute class weights for balanced sampling
def get_class_weights(labels):
    counts = np.bincount(labels)
    total = len(labels)
    return total / (len(counts) * counts)

# Train for one epoch
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        for i in range(3):
            mask = (labels == i)
            class_correct[i] += ((preds == labels) & mask).sum().item()
            class_total[i] += mask.sum().item()
    acc = correct / total
    class_acc = np.array([class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(3)])
    return total_loss / len(loader), acc, class_acc

# Evaluate on validation/test set
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            for i in range(3):
                mask = (labels == i)
                class_correct[i] += ((preds == labels) & mask).sum().item()
                class_total[i] += mask.sum().item()
    acc = correct / total
    class_acc = np.array([class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(3)])
    return total_loss / len(loader), acc, class_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)

# Plot boxplots of class accuracies across folds
def plot_fold_boxplots(fold_results, class_accuracies, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    class_names = ['Clean', 'EOG', 'EMG']
    acc_data = []
    labels = []
    for i, name in enumerate(class_names):
        acc_data.extend(class_accuracies[i])
        labels.extend([name] * len(class_accuracies[i]))
    df = pd.DataFrame({'Class': labels, 'Accuracy': acc_data})
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Accuracy', data=df)
    plt.title('ResNet1D_Model Accuracy Distribution by Class')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.3)
    plt.ylim(0.6, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ResNet1D_Model_accuracy_boxplot.png'))
    plt.close()

# Plot ROC curves for all folds
def plot_roc_curves(fold_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    class_names = ['Clean', 'EOG', 'EMG']
    colors = ['b', 'g', 'r']
    auc_vals = {name: [] for name in class_names}
    for idx, res in enumerate(fold_results):
        probs = res['probabilities']
        labels = res['labels']
        for i, name in enumerate(class_names):
            bin_labels = (labels == i).astype(int)
            class_probs = probs[:, i]
            fpr, tpr, _ = roc_curve(bin_labels, class_probs)
            roc_auc = auc(fpr, tpr)
            auc_vals[name].append(roc_auc)
            plt.plot(fpr, tpr, color=colors[i], alpha=0.1)
    # Plot mean ROC
    for i, name in enumerate(class_names):
        mean_auc = np.mean(auc_vals[name])
        std_auc = np.std(auc_vals[name])
        plt.plot([], [], color=colors[i], alpha=0.8,
                 label=f'{name} AUC = {mean_auc:.3f} ± {std_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ResNet1D_Model ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ResNet1D_Model_roc_curves.png'))
    plt.close()

# Plot aggregated results including confusion matrix and class performance
def plot_aggregated_results(fold_results, save_dir, class_accuracies):
    os.makedirs(save_dir, exist_ok=True)
    all_preds = np.concatenate([res['predictions'] for res in fold_results])
    all_labels = np.concatenate([res['labels'] for res in fold_results])
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
    class_acc_mean = np.mean(list(class_accuracies.values()), axis=1) * 100
    class_acc_std = np.std(list(class_accuracies.values()), axis=1) * 100
    summary_path = os.path.join(save_dir, 'ResNet1D_Model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('ResNet1D_Model Performance Summary\n')
        f.write('=======================================\n\n')
        f.write('Overall Performance:\n\n')
        avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
        std_acc = np.std([r['best_val_acc'] for r in fold_results])
        f.write(f'Overall Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n\n')
        f.write('Class-wise Performance:\n')
        for idx, name in enumerate(['Clean', 'EOG', 'EMG']):
            f.write(f'{name}: {class_acc_mean[idx]:.4f} ± {class_acc_std[idx]:.4f}\n')
        f.write('\nAverage Confusion Matrix (%):\n')
        f.write(np.array2string(cm_norm, precision=1, suppress_small=True, floatmode='fixed'))
        # Write classification report and aggregated metrics
        report = classification_report(all_labels, all_preds, target_names=['Clean', 'EOG', 'EMG'])
        f.write('\nClassification Report:\n')
        f.write(report + '\n')
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro')
        rec = recall_score(all_labels, all_preds, average='macro')
        f.write('Aggregated Metrics:\n')
        f.write(f'Overall Accuracy: {acc:.4f}\n')
        f.write(f'Macro Precision: {prec:.4f}\n')
        f.write(f'Macro Recall: {rec:.4f}\n')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm_norm, fmt='.1f', cmap='Blues')
    plt.title('ResNet1D_Model Avg Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.tight_layout()  # Commented out to match spacing of other models
    plt.savefig(os.path.join(save_dir, 'ResNet1D_Model_avg_confusion_matrix.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    x = np.arange(3)
    plt.bar(x, class_acc_mean, yerr=class_acc_std, capsize=5, color='skyblue')
    plt.xticks(x, ['Clean', 'EOG', 'EMG'])
    plt.ylabel('Accuracy (%)')
    plt.title('ResNet1D_Model Class-wise Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ResNet1D_Model_class_performance.png'))
    plt.close()
    plot_fold_boxplots(fold_results, class_accuracies, save_dir)
    plot_roc_curves(fold_results, save_dir)

    # t-SNE of aggregated predicted probabilities
    all_probs = np.concatenate([res['probabilities'] for res in fold_results])
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_probs)
    plt.figure(figsize=(10, 6))
    # scatter per class
    class_names = ['Clean', 'EOG', 'EMG']
    colors = ['b', 'g', 'r']
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_labels == cls_idx
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                    c=colors[cls_idx], label=cls_name, alpha=0.7)
    plt.title('ResNet1D_Model t-SNE of Predicted Probabilities')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ResNet1D_Model_tsne.png'))
    plt.close()

# Entry point
def main():
    results_dir = '/root/autodl-tmp/DeepSeparator-main/results/resnet1d_model'
    os.makedirs(results_dir, exist_ok=True)
    signals = np.load('classification_signals.npy')
    labels = np.load('classification_labels.npy')
    signals = torch.FloatTensor(signals)
    labels = torch.LongTensor(labels)
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.0005
    num_folds = 10
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    class_accuracies = {i: [] for i in range(3)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold, (train_idx, test_idx) in enumerate(skf.split(signals, labels)):
        print(f'Fold {fold+1}/{num_folds}')
        train_s, train_l = signals[train_idx], labels[train_idx]
        test_s, test_l = signals[test_idx], labels[test_idx]
        train_np, test_np = standardize_data(train_s.numpy(), test_s.numpy())
        train_s = torch.FloatTensor(train_np)
        test_s = torch.FloatTensor(test_np)
        class_w = get_class_weights(train_l.numpy())
        samp_w = class_w[train_l.numpy()]
        sampler = WeightedRandomSampler(weights=samp_w, num_samples=len(samp_w), replacement=True)
        train_loader = DataLoader(TensorDataset(train_s, train_l), batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(TensorDataset(test_s, test_l), batch_size=batch_size, shuffle=False)
        model = EEGResNet().to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_w).to(device))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_val_acc = 0.0
        min_delta = 1e-3  # minimum improvement to reset early stopping
        patience = 10     # increase patience to allow more epochs
        counter = 0
        for epoch in range(num_epochs):
            train_loss, train_acc, train_ca = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_ca, val_preds, val_labels, val_probs = evaluate(model, test_loader, criterion, device)
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_preds, best_labels, best_probs = val_preds, val_labels, val_probs
                counter = 0
            else:
                counter += 1
            for i in range(3):
                class_accuracies[i].append(val_ca[i])
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement > {min_delta} for {patience} epochs)')
                break
        fold_results.append({
            'fold': fold+1,
            'best_val_acc': best_val_acc,
            'predictions': best_preds,
            'labels': best_labels,
            'probabilities': best_probs
        })
    plot_aggregated_results(fold_results, os.path.join(results_dir, 'aggregated'), class_accuracies)
    avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_acc = np.std([r['best_val_acc'] for r in fold_results])
    print('\nResNet1D_Model Performance Summary')
    print(f'Overall Accuracy: {avg_acc:.4f} ± {std_acc:.4f}')
    for i, name in enumerate(['Clean','EOG','EMG']):
        mean_acc = np.mean(class_accuracies[i]) * 100
        std_acc_cls = np.std(class_accuracies[i]) * 100
        print(f'{name}: {mean_acc:.2f}% ± {std_acc_cls:.2f}%')
    cm = sum(confusion_matrix(r['labels'], r['predictions']) for r in fold_results)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
    print('\nAvg Confusion Matrix (%):')
    print(np.array2string(cm_norm, precision=1, suppress_small=True, floatmode='fixed'))

if __name__ == '__main__':
    main() 