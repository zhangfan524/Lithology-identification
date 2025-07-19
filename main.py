import os
import argparse
import torch
import shutil
from datetime import datetime

import config
from train import train_model
from utils import compare_models


def clean_empty_dirs(output_dir):
    """Clean up the empty output directory"""
    if not os.path.exists(output_dir):
        return
        
    for item in os.listdir(output_dir):
        dir_path = os.path.join(output_dir, item)
        if os.path.isdir(dir_path):
            
            is_empty = True
            for root, dirs, files in os.walk(dir_path):
                if files:
                    is_empty = False
                    break
            
            if is_empty:
                shutil.rmtree(dir_path)
                print(f"Empty directories have been deleted: {dir_path}")


def main():
    """
    The main function trains and evaluates the model based on command-line arguments
    """
    parser = argparse.ArgumentParser(description='Rock image classification')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'squeezenet', 'shufflenetv2', 'mobilenetv3', 'all'],
                        help='Model name')
    parser.add_argument('--attention', type=str, default=config.DEFAULT_ATTENTION,
                        choices=config.ATTENTION_TYPES,
                        help='Type of attention mechanism: cbam-CBAM attention mechanism, se-SE attention mechanism, eca-ECA attention mechanism, spatial-attention mechanism, channel-attention mechanism, none-attention mechanism')
    parser.add_argument('--attention-ablation', action='store_true', 
                        help='An ablation study of attention mechanisms will be conducted, where the same model will be trained using all types of attention mechanisms.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--clean', action='store_true', help='Clean up the empty output directory')
    
    args = parser.parse_args()
    
    
    if args.clean:
        clean_empty_dirs(config.OUTPUT_DIR)
        print("The cleanup of empty directories has been completed")
        return
    
    
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
        print(f"Set the number of training epochs: {config.NUM_EPOCHS}")
    
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        print(f"Set the batch size: {config.BATCH_SIZE}")
    
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
        print(f"Set the learning rate: {config.LEARNING_RATE}")
    
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_output_dir = os.path.join(config.OUTPUT_DIR, timestamp)
        os.makedirs(run_output_dir, exist_ok=True)
        
        
        with open(os.path.join(run_output_dir, 'config.txt'), 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Attention mechanism: {args.attention}\n")
            f.write(f"Batch size: {config.BATCH_SIZE}\n")
            f.write(f"Number of training epochs: {config.NUM_EPOCHS}\n")
            f.write(f"Learning rate: {config.LEARNING_RATE}\n")
            f.write(f"Weight decay: {config.WEIGHT_DECAY}\n")
            f.write(f"Image size: {config.IMG_SIZE}\n")
        
        print(f"Output directory: {run_output_dir}")
        
        
        if args.attention_ablation:
            print(f"\nPerform an attention mechanism ablation study: Model {args.model}")
            models_metrics = {}
            
            for attention_type in config.ATTENTION_TYPES:
                model_with_attn_name = f"{args.model}_{attention_type}"
                print(f"\nTrain the model: {args.model}, Attention mechanism: {attention_type}")
                metrics = train_model(args.model, attention_type, run_output_dir)
                models_metrics[model_with_attn_name] = metrics
            
            
            compare_models(models_metrics, run_output_dir)
            
        
        elif args.model == 'all':
            print("Train all models")
            models_metrics = {}
            
            for model_name in config.MODEL_NAMES:
                print(f"\nStart training the model: {model_name}, Attention mechanism: {args.attention}")
                metrics = train_model(model_name, args.attention, run_output_dir)
                models_metrics[model_name] = metrics
            
           
            compare_models(models_metrics, run_output_dir)
        else:
            print(f"\nStart training the model: {args.model}, Attention mechanism: {args.attention}")
            train_model(args.model, args.attention, run_output_dir)
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
        
        if 'run_output_dir' in locals():
            if os.path.exists(run_output_dir):
                if not os.listdir(run_output_dir):
                    os.rmdir(run_output_dir)
                    print(f"Empty output directories have been deleted: {run_output_dir}")
        raise  


if __name__ == '__main__':
    main() 