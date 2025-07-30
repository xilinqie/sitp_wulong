import torch
import argparse
from pathlib import Path

# 导入自定义模块
from model import DragonSegmentationHybrid
from trainer import HybridDragonTrainer
from dataset import create_data_loaders

def main():
    parser = argparse.ArgumentParser(description='Train Dragon Segmentation Model')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--sequence_length', type=int, default=8, help='Video sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_epochs': args.max_epochs,
        'save_dir': args.save_dir,
        'use_wandb': args.use_wandb,
        'phase1_epochs': 20,
        'phase2_epochs': 30,
        'phase3_epochs': 50,
        'dice_weight': 1.0,
        'focal_weight': 1.0,
        'temporal_weight': 0.5,
        'edge_weight': 0.3,
        'motion_weight': 0.2,
        'weight_decay': 1e-2,
        'min_lr': 1e-6
    }
    
    # 创建模型
    model = DragonSegmentationHybrid(
        sam2_model_name="facebook/sam2-hiera-large",
        input_size=(1024, 1024),
        num_frames=args.sequence_length,
        mamba_dim=256
    )
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length
    )
    
    # 创建训练器
    trainer = HybridDragonTrainer(model, config)
    
    # 开始训练
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")

if __name__ == '__main__':
    main()