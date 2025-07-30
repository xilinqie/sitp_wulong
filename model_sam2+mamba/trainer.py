import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import os
from pathlib import Path

class HybridDragonTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 损失函数
        from loss import DragonSegmentationLoss
        self.criterion = DragonSegmentationLoss(
            dice_weight=config.get('dice_weight', 1.0),
            focal_weight=config.get('focal_weight', 1.0),
            temporal_weight=config.get('temporal_weight', 0.5),
            edge_weight=config.get('edge_weight', 0.3),
            motion_weight=config.get('motion_weight', 0.2)
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-2)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 分阶段训练策略
        self.training_phases = {
            'phase1': self._phase1_sam2_adaptation,
            'phase2': self._phase2_mamba_training, 
            'phase3': self._phase3_joint_finetuning
        }
        
        # 当前训练阶段
        self.current_phase = 'phase1'
        self.phase_epochs = {
            'phase1': config.get('phase1_epochs', 20),
            'phase2': config.get('phase2_epochs', 30),
            'phase3': config.get('phase3_epochs', 50)
        }
        
    def _phase1_sam2_adaptation(self):
        """阶段1: SAM2适应性训练"""
        print("Phase 1: SAM2 Adaptation Training")
        # 只训练SAM2的最后几层和最终解码器
        for name, param in self.model.named_parameters():
            if ('sam2' in name and ('blocks.10' in name or 'blocks.11' in name)) or 'final_decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def _phase2_mamba_training(self):
        """阶段2: Mamba组件训练"""
        print("Phase 2: Mamba Component Training")
        # 冻结SAM2，训练Mamba部分
        for name, param in self.model.named_parameters():
            if 'mamba' in name or 'dragon' in name or 'fusion' in name or 'final_decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def _phase3_joint_finetuning(self):
        """阶段3: 联合微调"""
        print("Phase 3: Joint Fine-tuning")
        # 端到端微调整个模型
        for param in self.model.parameters():
            param.requires_grad = True
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        best_val_loss = float('inf')
        total_epochs = 0
        
        # 初始化wandb
        if self.config.get('use_wandb', False):
            wandb.init(project="dragon-segmentation", config=self.config)
        
        for phase_name, phase_epochs in self.phase_epochs.items():
            print(f"\n=== Starting {phase_name} ===")
            
            # 设置训练阶段
            self.current_phase = phase_name
            self.training_phases[phase_name]()
            
            # 训练该阶段
            for epoch in range(phase_epochs):
                total_epochs += 1
                
                # 训练一个epoch
                train_loss = self._train_epoch(train_loader, total_epochs)
                
                # 验证
                val_loss = self._validate_epoch(val_loader, total_epochs)
                
                # 学习率调度
                self.scheduler.step()
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(total_epochs, is_best=True)
                
                # 定期保存
                if total_epochs % 10 == 0:
                    self._save_checkpoint(total_epochs, is_best=False)
                
                print(f"Epoch {total_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # 记录到wandb
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'epoch': total_epochs,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'phase': phase_name
                    })
    
    def _train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            images = batch['images'].to(self.device)  # (B, T, C, H, W)
            masks = batch['masks'].to(self.device)    # (B, T, H, W)
            
            # 前向传播
            with autocast():
                predictions = self.model(images)
                
                # 计算损失
                targets = {'masks': masks}
                loss, loss_dict = self.criterion(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
            
            for batch in pbar:
                # 数据移动到设备
                images = batch['images'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # 前向传播
                with autocast():
                    predictions = self.model(images)
                    
                    # 计算损失
                    targets = {'masks': masks}
                    loss, loss_dict = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'val_loss': f"{total_loss/num_batches:.4f}"})
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'phase': self.current_phase,
            'config': self.config
        }
        
        # 创建保存目录
        save_dir = Path(self.config.get('save_dir', './checkpoints'))
        save_dir.mkdir(exist_ok=True)
        
        # 保存检查点
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")