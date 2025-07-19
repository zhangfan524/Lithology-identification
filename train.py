import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import config
from models import get_model
from utils import get_data_loaders, save_checkpoint, plot_metrics, save_metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, scaler=None):
    """
    训练模型一个epoch
    
    Args:
        model: Model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        scheduler: Learning rate scheduler (if using OneCycleLR, update every batch)
        scaler: Gradient scaler for mixed-precision training
        
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device, non_blocking=config.NON_BLOCKING), targets.to(device, non_blocking=config.NON_BLOCKING)
        
        
        optimizer.zero_grad()
        
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, scaler=None):
    """
    Evaluate the model on the validation set
    
    Args:
        model: Model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        scaler: Gradient scaler for mixed-precision training
        
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
        recall: Recall
        precision: Precision
        f1: F1 score
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device, non_blocking=config.NON_BLOCKING), targets.to(device, non_blocking=config.NON_BLOCKING)
            
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, recall, precision, f1


def test_per_class(model, test_loader, criterion, device, class_names, scaler=None):
    """
    Evaluate the model on the test set and calculate loss for each class
    
    Args:
        model: Model
        test_loader: Test data loader
        criterion: Loss function
        device: Device
        class_names: List of class names
        scaler: Gradient scaler for mixed-precision training
        
    Returns:
        class_losses: Dictionary containing average loss for each class
    """
    model.eval()
    class_losses = {name: 0.0 for name in class_names}
    class_counts = {name: 0 for name in class_names}
    
    criterion_none = nn.CrossEntropyLoss(reduction='none') # To get per-sample loss
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Calculating per-class losses"):
            inputs, targets = inputs.to(device, non_blocking=config.NON_BLOCKING), targets.to(device, non_blocking=config.NON_BLOCKING)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    losses = criterion_none(outputs, targets)
            else:
                outputs = model(inputs)
                losses = criterion_none(outputs, targets)

            for i in range(len(targets)):
                class_idx = targets[i].item()
                class_name = class_names[class_idx]
                class_losses[class_name] += losses[i].item()
                class_counts[class_name] += 1

    avg_class_losses = {name: (loss / class_counts[name]) if class_counts[name] > 0 else 0.0 
                        for name, loss in class_losses.items()}

    return avg_class_losses


def save_best_test_losses(class_losses, model_name, output_dir):
    """
    Save the best loss rate for each class on the test set to a txt file
    
    Args:
        class_losses: Dictionary containing losses for each class
        model_name: Model name
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f'best_test_losses_{model_name}_{timestamp}.txt')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Best loss rates for each class on test set - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for class_name, loss in class_losses.items():
            f.write(f"{class_name}: {loss:.4f}\n")
            
    print(f"Best test losses for each class saved to：{log_path}")


def save_training_data(train_losses, train_accs, val_losses, val_accs, val_recalls, val_precisions, val_f1s, model_name, output_dir):
    """
    Save training data to a txt file, grouped by category
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        val_recalls: List of validation recalls
        val_precisions: List of validation precisions
        val_f1s: List of validation F1 scores
        model_name: Model name
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f'training_data_{model_name}_{timestamp}.txt')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Model training data record - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        
        f.write("Training losses:\n")
        f.write("-" * 30 + "\n")
        for i, loss in enumerate(train_losses, 1):
            f.write(f"Epoch {i}: {loss:.4f}\n")
        f.write("\n")
        
        
        f.write("Training accuracies:\n")
        f.write("-" * 30 + "\n")
        for i, acc in enumerate(train_accs, 1):
            f.write(f"Epoch {i}: {acc*100:.2f}%\n")
        f.write("\n")
        
        
        f.write("Validation losses:\n")
        f.write("-" * 30 + "\n")
        for i, loss in enumerate(val_losses, 1):
            f.write(f"Epoch {i}: {loss:.4f}\n")
        f.write("\n")
        
        
        f.write("Validation accuracies:\n")
        f.write("-" * 30 + "\n")
        for i, acc in enumerate(val_accs, 1):
            f.write(f"Epoch {i}: {acc*100:.2f}%\n")
        f.write("\n")
        
        
        f.write("Validation recalls:\n")
        f.write("-" * 30 + "\n")
        for i, recall in enumerate(val_recalls, 1):
            f.write(f"Epoch {i}: {recall*100:.2f}%\n")
        f.write("\n")
        
        
        f.write("Validation precisions:\n")
        f.write("-" * 30 + "\n")
        for i, precision in enumerate(val_precisions, 1):
            f.write(f"Epoch {i}: {precision*100:.2f}%\n")
        f.write("\n")
        
        
        f.write("Validation F1 scores:\n")
        f.write("-" * 30 + "\n")
        for i, f1 in enumerate(val_f1s, 1):
            f.write(f"Epoch {i}: {f1*100:.2f}%\n")
        f.write("\n")
        
        
        f.write("Best results:\n")
        f.write("-" * 30 + "\n")
        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1
        f.write(f"Best validation accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})\n")
        f.write(f"Corresponding training accuracy: {train_accs[best_epoch-1]*100:.2f}%\n")
        f.write(f"Corresponding training loss: {train_losses[best_epoch-1]:.4f}\n")
        f.write(f"Corresponding validation loss: {val_losses[best_epoch-1]:.4f}\n")
        f.write(f"Corresponding validation recall: {val_recalls[best_epoch-1]*100:.2f}%\n")
        f.write(f"Corresponding validation precision: {val_precisions[best_epoch-1]*100:.2f}%\n")
        f.write(f"Corresponding validation F1 score: {val_f1s[best_epoch-1]*100:.2f}%\n")
    
    print(f"Training data saved to：{log_path}")


def train_model(model_name, attention_type=config.DEFAULT_ATTENTION, output_dir=None):
    """
    Train and evaluate the model
    
    Args:
        model_name: Model name
        attention_type: Type of attention mechanism
        output_dir: Output directory, create new directory if None
        
    Returns:
        metrics: Output directory, create new directory if None
    """
    
    if config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif config.DEVICE == 'dml':
        try:
            device = torch_directml.device()
            print(f"Using DirectML device: {device}")
        except (NameError, ImportError):
            print("Warning: torch_directml not imported correctly, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Training with CPU")
    
    
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        models_dir = os.path.join(output_dir, 'models')
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    else:
        models_dir = os.path.join(output_dir, 'models')
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    
    
    train_loader, test_loader, num_classes = get_data_loaders()
    print(f"Number of classes: {num_classes}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    
    model = get_model(model_name, num_classes, attention_type)
    model = model.to(device)
    
    
    if hasattr(config, 'OPTIMIZER'):
        if config.OPTIMIZER.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config.LEARNING_RATE, 
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config.LEARNING_RATE, 
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=config.LEARNING_RATE, 
                momentum=0.9, 
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config.LEARNING_RATE, 
                weight_decay=config.WEIGHT_DECAY
            )
    else:
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
    
    
    if hasattr(config, 'LR_SCHEDULER'):
        if config.LR_SCHEDULER.lower() == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=config.NUM_EPOCHS
            )
        elif config.LR_SCHEDULER.lower() == 'onecyclelr':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config.LEARNING_RATE,
                steps_per_epoch=len(train_loader),
                epochs=config.NUM_EPOCHS,
                pct_start=getattr(config, 'PCT_START', 0.3)
            )
        elif config.LR_SCHEDULER.lower() == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=5, 
                verbose=True
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    
    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
  
    scaler = None
    if hasattr(config, 'MIXED_PRECISION') and config.MIXED_PRECISION and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed-precision training enabled")
    
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs, val_recalls, val_precisions, val_f1s = [], [], [], [], []
    
    
    model_save_name = f"{model_name}_{attention_type}"
    model_save_path = os.path.join(models_dir, f"{model_save_name}_best.pth")
    
    
    print(f"Starting training: {model_save_name}")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        
        train_loss, train_acc = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            scheduler if isinstance(scheduler, OneCycleLR) else None,
            scaler
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        
        val_loss, val_acc, val_recall, val_precision, val_f1 = validate(model, test_loader, criterion, device, scaler)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)
        val_f1s.append(val_f1)
        
        
        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        
        print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc*100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc*100:.2f}%, Validation Recall: {val_recall:.4f}, Validation Precision: {val_precision:.4f}")
        
       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'attention_type': attention_type
            }, model_save_path)
            print(f"Saving best model, validation accuracy: {best_val_acc*100:.2f}%")
        
        
        if hasattr(config, 'SAVE_FREQ') and config.SAVE_FREQ > 0 and (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(models_dir, f"{model_save_name}_epoch{epoch+1}.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'attention_type': attention_type
            }, checkpoint_path)
            print(f"Saved model at epoch {epoch+1}")
    
    
    plot_metrics(train_losses, train_accs, val_losses, val_accs, model_save_name, plots_dir)
    
    
    metrics = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_recalls': val_recalls,
        'val_precisions': val_precisions,
        'val_f1s': val_f1s,
        'best_val_acc': best_val_acc
    }
    metrics_path = os.path.join(plots_dir, f"{model_save_name}_metrics.pth")
    save_metrics(metrics, metrics_path)
    
    
    save_training_data(train_losses, train_accs, val_losses, val_accs, val_recalls, val_precisions, val_f1s, model_save_name, output_dir)
    
    

    
    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=config.NON_BLOCKING)
            targets = targets.to(device, non_blocking=config.NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    class_names = train_loader.dataset.classes

    
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(class_names)
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    recall_per_class_percent = recall_per_class * 100
    cm_with_acc = np.concatenate([cm, recall_per_class_percent.reshape(-1, 1)], axis=1)
    class_names_with_acc = class_names + ['acc(%)']

    
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_sum, where=cm_sum!=0) * 100
    labels = np.empty_like(cm).astype(str)
    for i in range(num_classes):
        for j in range(num_classes):
            labels[i, j] = f"{cm[i, j]}\n{cm_perc[i, j]:.2f}%"
    acc_labels = np.array([f"{recall_per_class_percent[i]:.2f}%" for i in range(num_classes)]).reshape(-1, 1)
    labels_with_acc = np.concatenate([labels, acc_labels], axis=1)

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_with_acc,
        annot=labels_with_acc,
        fmt='',
        cmap='Blues',
        xticklabels=class_names_with_acc,
        yticklabels=class_names
    )
    plt.xlabel('Prediction label')
    plt.ylabel('Real label')
    plt.title(f'confusion matrix of {model_save_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{model_save_name}_confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved to: {os.path.join(plots_dir, f'{model_save_name}_confusion_matrix.png')}")

   
    class_losses = test_per_class(model, test_loader, criterion, device, class_names, scaler)
    save_best_test_losses(class_losses, model_save_name, output_dir)

    print(f"\nTraining completed: {model_save_name}")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

    return metrics