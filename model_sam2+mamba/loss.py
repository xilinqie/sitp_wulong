import torch
import torch.nn as nn
import torch.nn.functional as F

class DragonSegmentationLoss(nn.Module):
    """舞龙分割专用损失函数"""
    
    def __init__(self, 
                 dice_weight=1.0,
                 focal_weight=1.0, 
                 temporal_weight=0.5,
                 edge_weight=0.3,
                 motion_weight=0.2):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.temporal_weight = temporal_weight
        self.edge_weight = edge_weight
        self.motion_weight = motion_weight
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.edge_loss = EdgeLoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型输出字典
            targets: 真实标签字典
        """
        pred_masks = predictions['masks']  # (B, T, C, H, W)
        true_masks = targets['masks']      # (B, T, H, W)
        
        total_loss = 0
        loss_dict = {}
        
        # 1. Dice Loss
        dice_loss = self.dice_loss(pred_masks, true_masks)
        total_loss += self.dice_weight * dice_loss
        loss_dict['dice_loss'] = dice_loss
        
        # 2. Focal Loss
        focal_loss = self.focal_loss(pred_masks, true_masks)
        total_loss += self.focal_weight * focal_loss
        loss_dict['focal_loss'] = focal_loss
        
        # 3. 时序一致性损失
        if pred_masks.shape[1] > 1:  # 多帧
            temporal_loss = self.temporal_consistency_loss(pred_masks)
            total_loss += self.temporal_weight * temporal_loss
            loss_dict['temporal_loss'] = temporal_loss
        
        # 4. 边缘损失
        edge_loss = self.edge_loss(pred_masks, true_masks)
        total_loss += self.edge_weight * edge_loss
        loss_dict['edge_loss'] = edge_loss
        
        # 5. 运动一致性损失
        if 'motion_flows' in predictions and len(predictions['motion_flows']) > 0:
            motion_loss = self.motion_consistency_loss(predictions['motion_flows'], pred_masks)
            total_loss += self.motion_weight * motion_loss
            loss_dict['motion_loss'] = motion_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
    
    def dice_loss(self, pred, target):
        """Dice损失"""
        smooth = 1e-5
        
        pred = torch.sigmoid(pred)
        
        # 处理维度差异：pred (B,T,C,H,W), target (B,T,H,W)
        if len(pred.shape) == 5 and len(target.shape) == 4:
            # 对于多类别，取第一个类别（dragon类）
            pred = pred[:, :, 1]  # 假设dragon是第1个类别
        
        # 如果pred和target尺寸不匹配，上采样pred到target的尺寸
        if pred.shape[-2:] != target.shape[-2:]:
            # 需要重新调整形状以适应F.interpolate
            B, T = pred.shape[:2]
            pred = pred.view(B*T, 1, pred.shape[-2], pred.shape[-1])
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            pred = pred.view(B, T, target.shape[-2], target.shape[-1])
        
        # 确保张量连续并展平
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def temporal_consistency_loss(self, pred_masks):
        """时序一致性损失"""
        B, T, C, H, W = pred_masks.shape
        
        if T < 2:
            return torch.tensor(0.0, device=pred_masks.device)
        
        consistency_loss = 0
        for t in range(1, T):
            # 计算相邻帧的差异
            diff = torch.abs(pred_masks[:, t] - pred_masks[:, t-1])
            consistency_loss += diff.mean()
        
        return consistency_loss / (T - 1)
    
    def motion_consistency_loss(self, motion_flows, pred_masks):
        """运动一致性损失"""
        # 简化的运动一致性计算
        motion_loss = 0
        for i, flow in enumerate(motion_flows):
            if flow is not None:
                # 基于光流的掩码变形
                warped_mask = self.warp_mask(pred_masks[:, i], flow)
                motion_loss += F.mse_loss(warped_mask, pred_masks[:, i+1])
        
        return motion_loss / len(motion_flows) if motion_flows else torch.tensor(0.0)
    
    def warp_mask(self, mask, flow):
        """基于光流变形掩码"""
        # 简化实现
        return mask  # 实际应该实现光流变形

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # 处理维度差异：pred (B,T,C,H,W), target (B,T,H,W)
        if len(pred.shape) == 5 and len(target.shape) == 4:
            # 对于多类别，取第一个类别（dragon类）
            pred = pred[:, :, 1]  # 假设dragon是第1个类别
        
        # 确保pred和target的空间分辨率一致
        if pred.shape[-2:] != target.shape[-2:]:
            # 上采样pred到target的分辨率
            B, T = pred.shape[:2]
            pred = pred.view(B*T, 1, pred.shape[-2], pred.shape[-1])
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            pred = pred.view(B, T, target.shape[-2], target.shape[-1])
        
        # 使用binary_cross_entropy_with_logits以支持autocast
        ce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        pred_sigmoid = torch.sigmoid(pred)
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        
        return loss.mean()

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # 处理维度差异：pred (B,T,C,H,W), target (B,T,H,W)
        if len(pred.shape) == 5 and len(target.shape) == 4:
            # 对于多类别，取第一个类别（dragon类）
            pred = pred[:, :, 1]  # 假设dragon是第1个类别
        
        # 确保pred和target的空间分辨率一致
        if pred.shape[-2:] != target.shape[-2:]:
            # 上采样pred到target的分辨率
            B, T = pred.shape[:2]
            pred = pred.view(B*T, 1, pred.shape[-2], pred.shape[-1])
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
            pred = pred.view(B, T, target.shape[-2], target.shape[-1])
        
        # 计算边缘
        pred_edges = self.get_edges(torch.sigmoid(pred))
        target_edges = self.get_edges(target.float())
        
        return F.mse_loss(pred_edges, target_edges)
    
    def get_edges(self, x):
        # Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        if x.dim() == 4:  # (B, T, H, W)
            x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        elif x.dim() == 3:  # (B, H, W)
            x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        
        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)
        
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        return edges