import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from attention_classifier import AttentionClassifier
import matplotlib.pyplot as plt
from evaluation_utils import ModelEvaluator
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.manifold import TSNE
import pandas as pd
import torch.nn.functional as F

def standardize_data(train_data, test_data):
    scaler = StandardScaler()
    train_shape = train_data.shape
    test_shape = test_data.shape
    
    train_data = train_data.reshape(-1, train_data.shape[-1])
    test_data = test_data.reshape(-1, test_data.shape[-1])
    
    scaler.fit(train_data)
    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    train_data = train_data.reshape(train_shape)
    test_data = test_data.reshape(test_shape)
    
    return train_data, test_data

def get_class_weights(labels):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return weights

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(3):
            mask = (labels == i)
            class_correct[i] += ((predicted == labels) & mask).sum().item()
            class_total[i] += mask.sum().item()
    
    class_acc = np.zeros(3)
    for i in range(3):
        if class_total[i] > 0:
            class_acc[i] = class_correct[i] / class_total[i]
    
    return total_loss / len(train_loader), correct / total, class_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            for i in range(3):
                mask = (labels == i)
                class_correct[i] += ((predicted == labels) & mask).sum().item()
                class_total[i] += mask.sum().item()
    
    class_acc = np.zeros(3)
    for i in range(3):
        if class_total[i] > 0:
            class_acc[i] = class_correct[i] / class_total[i]
    
    return (total_loss / len(test_loader), correct / total, class_acc,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))

def plot_fold_boxplots(fold_results, class_accuracies, save_dir):
    """Plot boxplots for accuracy distribution across folds"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for boxplots
    class_names = ['Clean', 'EOG', 'EMG']
    accuracies_data = []
    labels = []
    
    # Collect accuracies for each class
    for i, name in enumerate(class_names):
        accuracies_data.extend(class_accuracies[i])
        labels.extend([name] * len(class_accuracies[i]))
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'Class': labels,
        'Accuracy': accuracies_data
    })
    
    # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Accuracy', data=df)
    plt.title('CNN_Attention_Model Accuracy Distribution by Class')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.6, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'CNN_Attention_Model_accuracy_boxplot.png'))
    plt.close()

def plot_roc_curves(fold_results, save_dir):
    """Plot ROC curves for all folds on one figure"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    class_names = ['Clean', 'EOG', 'EMG']
    colors = ['b', 'g', 'r']
    
    # Store AUC values for each class and fold
    auc_values = {class_name: [] for class_name in class_names}
    
    # Plot ROC curve for each fold
    for fold_idx, fold_data in enumerate(fold_results):
        probabilities = fold_data['probabilities']
        labels = fold_data['labels']
        
        # Calculate ROC curve for each class
        for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
            # Prepare binary labels for current class
            binary_labels = (labels == class_idx).astype(int)
            class_probs = probabilities[:, class_idx]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            roc_auc = auc(fpr, tpr)
            auc_values[class_name].append(roc_auc)
            
            # Plot with low alpha for individual fold curves
            if fold_idx == 0:  # Only add label for first fold to avoid duplicate legend entries
                plt.plot(fpr, tpr, color=color, alpha=0.1,
                        label=f'{class_name} (individual folds)')
            else:
                plt.plot(fpr, tpr, color=color, alpha=0.1)
    
    # Plot mean ROC curves
    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        mean_auc = np.mean(auc_values[class_name])
        std_auc = np.std(auc_values[class_name])
        plt.plot([], [], color=color, alpha=0.8, linestyle='-',
                label=f'{class_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN_Attention_Model ROC Curves (All Folds)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'CNN_Attention_Model_roc_curves.png'))
    plt.close()

def plot_aggregated_results(fold_results, save_dir, class_accuracies):
    """Plot aggregated results across all folds"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Aggregate predictions and labels
    all_preds = np.concatenate([res['predictions'] for res in fold_results])
    all_labels = np.concatenate([res['labels'] for res in fold_results])
    
    # Calculate average confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentage
    
    # Calculate class-wise accuracies mean and std
    class_acc_mean = np.mean(list(class_accuracies.values()), axis=1) * 100
    class_acc_std = np.std(list(class_accuracies.values()), axis=1) * 100
    
    # Save performance summary
    with open(os.path.join(save_dir, 'CNN_Attention_Model_summary.txt'), 'w') as f:
        f.write("CNN_Attention_Model Performance Summary\n")
        f.write("==================================================\n\n")
        
        # Overall Performance
        avg_acc = np.mean([result['best_val_acc'] for result in fold_results])
        std_acc = np.std([result['best_val_acc'] for result in fold_results])
        f.write(f"Overall Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n\n")
        
        # Class-wise Performance
        f.write("Class-wise Performance:\n")
        class_names = ['Clean', 'EOG', 'EMG']
        for i, name in enumerate(class_names):
            f.write(f"{name:>8}: {class_acc_mean[i]:.4f} ± {class_acc_std[i]:.4f}\n")
        f.write("\n")
        
        # Confusion Matrix
        f.write("Average Confusion Matrix (%):\n")
        f.write(np.array2string(cm_normalized, precision=1, suppress_small=True, floatmode='fixed'))
        # Write classification report and aggregated metrics
        report = classification_report(all_labels, all_preds, target_names=class_names)
        f.write("\nClassification Report:\n")
        f.write(report + "\n")
        # Aggregated metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro')
        rec = recall_score(all_labels, all_preds, average='macro')
        f.write("Aggregated Metrics:\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {prec:.4f}\n")
        f.write(f"Macro Recall: {rec:.4f}\n")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=cm_normalized, fmt='.1f', cmap='Blues')
    plt.title('CNN_Attention_Model Average Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'CNN_Attention_Model_avg_confusion_matrix.png'))
    plt.close()
    
    # Plot class performance
    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x, class_acc_mean, width, yerr=class_acc_std, capsize=5,
           label='Accuracy', color='skyblue')
    
    plt.title('CNN_Attention_Model Class-wise Performance')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, class_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'CNN_Attention_Model_class_performance.png'))
    plt.close()
    
    # Add new visualization plots
    plot_fold_boxplots(fold_results, class_accuracies, save_dir)
    plot_roc_curves(fold_results, save_dir)
    # t-SNE of aggregated predicted probabilities
    all_probs = np.concatenate([res['probabilities'] for res in fold_results])
    all_labels = np.concatenate([res['labels'] for res in fold_results])
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_probs)
    plt.figure(figsize=(10, 6))
    # scatter per class for t-SNE
    class_names = ['Clean', 'EOG', 'EMG']
    colors = ['b', 'g', 'r']
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_labels == cls_idx
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                    c=colors[cls_idx], label=cls_name, alpha=0.7)
    plt.title('CNN_Attention_Model t-SNE of Predicted Probabilities')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'CNN_Attention_Model_tsne.png'))
    plt.close()

class FocalLoss(nn.Module):
    """Focal Loss for classification tasks"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, inputs, targets):
        # Compute standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    # Create results directory
    results_dir = '/root/autodl-tmp/DeepSeparator-main/results/attention_model'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    signals = np.load('classification_signals.npy')
    labels = np.load('classification_labels.npy')
    
    # Convert to torch tensors
    signals = torch.FloatTensor(signals)
    labels = torch.LongTensor(labels)
    
    # Training parameters
    num_epochs = 100  # increased training epochs
    batch_size = 32
    learning_rate = 0.0005
    num_folds = 10
    
    # Initialize stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Store results
    fold_results = []
    class_accuracies = {i: [] for i in range(3)}
    
    # Cross validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(signals, labels)):
        print(f'Fold {fold + 1}/{num_folds}')
        
        # Split data
        train_signals = signals[train_idx]
        train_labels = labels[train_idx]
        test_signals = signals[test_idx]
        test_labels = labels[test_idx]
        
        # Standardize data for this fold
        train_signals_np, test_signals_np = standardize_data(
            train_signals.numpy(), 
            test_signals.numpy()
        )
        train_signals = torch.FloatTensor(train_signals_np)
        test_signals = torch.FloatTensor(test_signals_np)
        
        # Calculate class weights for balanced training
        class_weights = get_class_weights(train_labels.numpy())
        sample_weights = class_weights[train_labels.numpy()]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_dataset = TensorDataset(train_signals, train_labels)
        test_dataset = TensorDataset(test_signals, test_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=sampler
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Initialize model, loss function and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AttentionClassifier().to(device)
        # Use Focal Loss to focus on hard-to-classify EOG/EMG samples
        criterion = FocalLoss(gamma=2.0, alpha=torch.FloatTensor(class_weights).to(device))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Training loop with improved early stopping
        best_val_acc = 0.0
        min_delta = 1e-3   # minimum improvement to reset early stopping
        patience = 10      # increased patience
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc, train_class_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_class_acc, val_preds, val_labels, val_probs = evaluate(
                model, test_loader, criterion, device
            )
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # consider significant improvement only
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_preds = val_preds
                best_labels = val_labels
                best_probs = val_probs
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'Train Class Acc: {train_class_acc}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'Val Class Acc: {val_class_acc}')
        
            # Store results
            fold_results.append({
                'fold': fold + 1,
                'best_val_acc': best_val_acc,
                'predictions': best_preds,
                'labels': best_labels,
                'probabilities': best_probs
            })
            
            # Store class accuracies
            for i in range(3):
                class_accuracies[i].append(val_class_acc[i])
    
    # Plot aggregated results
    plot_aggregated_results(fold_results, os.path.join(results_dir, 'aggregated'), class_accuracies)
    
    # Calculate and print average performance
    avg_acc = np.mean([result['best_val_acc'] for result in fold_results])
    std_acc = np.std([result['best_val_acc'] for result in fold_results])
    print("\nCNN_Model Performance Summary")
    print("==================================================")
    print("\nOverall Performance:")
    print(f"Overall Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    
    print("\nClass-wise Performance:")
    class_names = ['Clean', 'EOG', 'EMG']
    class_acc_mean = np.mean(list(class_accuracies.values()), axis=1) * 100
    class_acc_std = np.std(list(class_accuracies.values()), axis=1) * 100
    for i, name in enumerate(class_names):
        print(f"{name:>8}: {class_acc_mean[i]:.4f} ± {class_acc_std[i]:.4f}")
    
    # Calculate and print confusion matrix
    all_preds = np.concatenate([res['predictions'] for res in fold_results])
    all_labels = np.concatenate([res['labels'] for res in fold_results])
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print("\nAverage Confusion Matrix (%):")
    print(np.array2string(cm_normalized, precision=1, suppress_small=True, floatmode='fixed'))

if __name__ == '__main__':
    main()