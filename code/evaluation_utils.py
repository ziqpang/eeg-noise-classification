import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

class ModelEvaluator:
    @staticmethod
    def plot_training_history(history, title, save_path):
        """Plot training and validation metrics history."""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{title} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title, save_path):
        """Plot confusion matrix."""
        cm = np.zeros((3, 3))
        for i in range(len(y_true)):
            cm[y_true[i]][y_pred[i]] += 1
            
        # Normalize confusion matrix to percentage
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=cm_normalized, fmt='.1f', cmap='Blues')
        plt.title(f'{title} (%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_roc_curves(y_true, y_prob, title, save_path):
        """Plot ROC curves for each class."""
        plt.figure(figsize=(10, 8))
        
        # Convert to one-hot encoding for ROC curve calculation
        y_true_onehot = np.zeros((len(y_true), 3))
        for i in range(len(y_true)):
            y_true_onehot[i][y_true[i]] = 1
        
        colors = ['blue', 'red', 'green']
        classes = ['Clean', 'EOG', 'EMG']
        
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i],
                     label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_class_accuracies(class_accuracies, title, save_path):
        """Plot class-wise accuracies across folds."""
        class_names = ['Clean', 'EOG', 'EMG']
        class_acc_mean = np.mean(list(class_accuracies.values()), axis=1) * 100
        class_acc_std = np.std(list(class_accuracies.values()), axis=1) * 100
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.bar(x, class_acc_mean, width, yerr=class_acc_std, capsize=5,
               label='Accuracy', color='skyblue')
        
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.xticks(x, class_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 