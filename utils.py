import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import config


def get_data_loaders():
    """
    Create training and testing data loaders
    
    Returns:
        train_loader: Training data loader
        test_loader: Testing data loader
        num_classes: Number of classes
    """
   
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(config.IMG_SIZE),  
            transforms.RandomCrop(config.CROP_SIZE),  
            transforms.RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP_PROB),
            transforms.RandomVerticalFlip(p=config.RANDOM_VERTICAL_FLIP_PROB),
            transforms.RandomRotation(config.RANDOM_ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER['brightness'],
                contrast=config.COLOR_JITTER['contrast'],
                saturation=config.COLOR_JITTER['saturation'],
                hue=config.COLOR_JITTER['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
            transforms.RandomErasing(
                p=config.RANDOM_ERASING_PROB,
                scale=config.RANDOM_ERASING_SCALE
            )
        ]),
        'test': transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.CROP_SIZE),  
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD)
        ]),
    }

    
    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=data_transforms['train']
    )
    
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=data_transforms['test']
    )
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    num_classes = len(train_dataset.classes)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {', '.join(train_dataset.classes)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    
    return train_loader, test_loader, num_classes


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model states
        filename: Save path
    """
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def plot_metrics(train_losses, train_accs, val_losses, val_accs, model_name, save_dir):
    """
    Plot loss and accuracy curves during training
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        model_name: Model name
        save_dir: Save directory
    """
    
    actual_epochs = len(train_losses)
    if actual_epochs == 0:
        print(f"Model {model_name} has no valid training data points, cannot generate charts.")
        return
    
    epochs_range = list(range(1, actual_epochs + 1))
    
    
    train_accs_percent = [acc * 100 for acc in train_accs]
    val_accs_percent = [acc * 100 for acc in val_accs]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    
    
    ax1.set_title(f'Model: {model_name} - Loss Curve', fontsize=14, weight='bold')
    ax1.set_ylabel('Loss (Cross Entropy)', fontsize=12)
    ax1.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    
    
    ax1.plot(epochs_range, train_losses, label='Training Loss', 
             color='#1f77b4', linestyle='-', linewidth=2, marker='.', markersize=6)
    
    
    ax1.plot(epochs_range, val_losses, label='Validation Loss', 
             color='#ff7f0e', linestyle='--', linewidth=2, marker='x', markersize=6)
    
    ax1.legend(loc='upper right', fontsize='medium')
    ax1.tick_params(axis='y', labelsize=10)
    
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    
    ax2.set_title(f'Model: {model_name} - Accuracy Curve', fontsize=14, weight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    
    
    ax2.plot(epochs_range, train_accs_percent, label='Training Accuracy', 
             color='#2ca02c', linestyle='-', linewidth=2, marker='.', markersize=6)
    
    
    ax2.plot(epochs_range, val_accs_percent, label='Validation Accuracy', 
             color='#d62728', linestyle='--', linewidth=2, marker='x', markersize=6)
    
    
    best_val_acc_idx = np.argmax(val_accs_percent)
    best_val_acc = val_accs_percent[best_val_acc_idx]
    best_epoch = best_val_acc_idx + 1
    
    
    ax2.scatter(best_epoch, best_val_acc, color='red', s=60, zorder=5,
               label=f'Best Validation Point (Epoch {best_epoch})',
               facecolors='none', edgecolors='red', linewidth=2)
    
    
    ax2.annotate(f'Best: {best_val_acc:.2f}%',
                xy=(best_epoch, best_val_acc),
                xytext=(best_epoch + 1, best_val_acc + 1),
                ha='left',
                fontsize=10,
                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=4, alpha=0.7))
    
    
    handles, labels = ax2.get_legend_handles_labels()
    filtered_handles = [h for h, l in zip(handles, labels) if 'Best Validation Point' in l or 'Accuracy' in l]
    filtered_labels = [l for l in labels if 'Best Validation Point' in l or 'Accuracy' in l]
    ax2.legend(filtered_handles, filtered_labels, loc='lower right', fontsize='medium')
    
    
    all_acc_data = train_accs_percent + val_accs_percent
    if all_acc_data:
        min_acc = np.min(all_acc_data)
        max_acc = np.max(all_acc_data)
        y_bottom = max(0, min_acc - 5)
        y_top = min(105, max_acc + 5)
        ax2.set_ylim(bottom=y_bottom, top=y_top)
    else:
        ax2.set_ylim(bottom=0, top=105)
    
    
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(axis='both', labelsize=10)
    
    
    fig.tight_layout(pad=1.5)
    save_path = os.path.join(save_dir, f'{model_name}_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"模型 {model_name} 的训练曲线图已保存至 {save_path}")
    plt.close(fig)


def compare_models(models_metrics, save_dir):
    """
    Compare performance of different models
    
    Args:
        models_metrics: Dictionary with model names as keys and metric dictionaries as values
        save_dir: Save directory
    """
   
    best_model = None
    best_acc = 0
    
    for model_name, metrics in models_metrics.items():
        
        val_accs_percent = [acc * 100 for acc in metrics['val_accs']]
        max_val_acc = max(val_accs_percent)
        
        
        if max_val_acc > best_acc:
            best_acc = max_val_acc
            best_model = model_name
    
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Models Comparison - Best: {best_model} ({best_acc:.2f}%)', 
                fontsize=16, weight='bold', y=0.98)
    
    
    max_epochs = 0
    for metrics in models_metrics.values():
        max_epochs = max(max_epochs, len(metrics['train_losses']))
    
    epochs_range = list(range(1, max_epochs + 1))
    
   
    ax1.set_title('Loss Comparison', fontsize=14, weight='bold')
    ax1.set_ylabel('Loss (Cross Entropy)', fontsize=12)
    ax1.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    
    for model_name, metrics in models_metrics.items():
        train_losses = metrics['train_losses']
        val_losses = metrics['val_losses']
        
       
        train_epochs = len(train_losses)
        val_epochs = len(val_losses)
        model_epochs = min(train_epochs, val_epochs)
        model_range = epochs_range[:model_epochs]
        
        
        ax1.plot(model_range, val_losses[:model_epochs], 
                label=f'{model_name} Validation',
                linewidth=2, marker='x', markersize=5)
    
    ax1.legend(loc='upper right', fontsize='medium')
    ax1.tick_params(axis='y', labelsize=10)
    
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    
    ax2.set_title('Accuracy Comparison', fontsize=14, weight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    
    best_points = {}  
    
    for model_name, metrics in models_metrics.items():
        train_accs = [acc * 100 for acc in metrics['train_accs']]
        val_accs = [acc * 100 for acc in metrics['val_accs']]
        
     
        train_epochs = len(train_accs)
        val_epochs = len(val_accs)
        model_epochs = min(train_epochs, val_epochs)
        model_range = epochs_range[:model_epochs]
        
        
        line, = ax2.plot(model_range, val_accs[:model_epochs], 
                        label=f'{model_name} Validation',
                        linewidth=2, marker='x', markersize=5)
        
        
        if val_accs:
            best_idx = np.argmax(val_accs[:model_epochs])
            best_acc = val_accs[best_idx]
            best_epoch = best_idx + 1
            
            
            best_points[model_name] = {
                'epoch': best_epoch,
                'accuracy': best_acc,
                'color': line.get_color()
            }
    
    
    for model_name, point_info in best_points.items():
        epoch = point_info['epoch']
        accuracy = point_info['accuracy']
        color = point_info['color']
        
        
        ax2.scatter(epoch, accuracy, color=color, s=80, zorder=5,
                   facecolors='none', edgecolors=color, linewidth=2)
        
        
        ax2.annotate(f'{model_name}: {accuracy:.2f}%',
                    xy=(epoch, accuracy),
                    xytext=(epoch + 0.5, accuracy + 1),
                    ha='left',
                    fontsize=9,
                    arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=4, alpha=0.7))
    
    
    all_accs = []
    for metrics in models_metrics.values():
        all_accs.extend([acc * 100 for acc in metrics['val_accs']])
    
    if all_accs:
        min_acc = np.min(all_accs)
        max_acc = np.max(all_accs)
        y_bottom = max(0, min_acc - 5)
        y_top = min(105, max_acc + 5)
        ax2.set_ylim(bottom=y_bottom, top=y_top)
    else:
        ax2.set_ylim(bottom=0, top=105)
    
    
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc='lower right', fontsize='medium')
    ax2.tick_params(axis='both', labelsize=10)
    
   
    fig.tight_layout(rect=[0, 0, 1, 0.95])  
    save_path = os.path.join(save_dir, 'models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Models comparison chart saved to {save_path}")
    plt.close(fig)
    
    print("\nFinal validation accuracies for each model:")
    for model_name, metrics in models_metrics.items():
        print(f"{model_name}: {metrics['val_accs'][-1]*100:.2f}%")
    
    
    print(f"\nBest model: {best_model}, highest validation accuracy: {best_acc:.2f}%")


def save_metrics(metrics, filename):
    """
    Save training metrics to file
    
    Args:
        metrics: Dictionary containing training metrics
        filename: Save path
    """
    torch.save(metrics, filename)
    print(f"Training metrics saved: {filename}")


def save_model(model, model_name, save_dir):
    """
     Save complete model (not just state dictionary)
    
    Args:
        model: Model
        model_name: Model name
        save_dir: Save directory
    """
    save_path = os.path.join(save_dir, f"{model_name}_full_model.pth")
    torch.save(model, save_path)
    print(f"Complete model saved: {save_path}") 