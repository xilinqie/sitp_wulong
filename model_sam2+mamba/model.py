import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM2Decoder(nn.Module):
    """SAM2解码器 - 生成初始分割掩码"""
    
    def __init__(self, embed_dim=256, num_multimask_outputs=3):
        super().__init__()
        
        self.mask_tokens = nn.Embedding(num_multimask_outputs + 1, embed_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(embed_dim, embed_dim, embed_dim // 8, 3) for _ in range(num_multimask_outputs + 1)
        ])
        
        self.iou_prediction_head = MLP(embed_dim, 256, 1, 3)
        
    def forward(self, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings):
        """解码生成分割掩码"""
        # 简化的SAM2解码逻辑
        batch_size = image_embeddings.shape[0]
        
        # 获取mask tokens
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 上采样特征
        upscaled_embedding = self.output_upscaling(image_embeddings)
        
        # 生成掩码
        masks = []
        iou_pred = []
        
        for i, mlp in enumerate(self.output_hypernetworks_mlps):
            mask_embedding = mlp(mask_tokens[:, i, :])
            mask = torch.einsum('bc,bchw->bhw', mask_embedding, upscaled_embedding)
            masks.append(mask.unsqueeze(1))
            
            # IoU预测
            iou = self.iou_prediction_head(mask_tokens[:, i, :])
            iou_pred.append(iou)
        
        masks = torch.cat(masks, dim=1)
        iou_pred = torch.cat(iou_pred, dim=1)
        
        return masks, iou_pred

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DragonSegmentationHybrid(nn.Module):
    """SAM2 + Mamba 混合舞龙分割模型"""
    
    def __init__(self, 
                 sam2_model_name="facebook/sam2-hiera-large",
                 input_size=(1024, 1024),
                 num_frames=8,
                 mamba_dim=256,
                 num_classes=4):
        super().__init__()
        
        self.input_size = input_size
        self.num_frames = num_frames
        self.mamba_dim = mamba_dim
        self.num_classes = num_classes
        
        # SAM2编码器 (简化版本)
        self.sam2_encoder = self._build_sam2_encoder()
        
        # 时序Mamba
        self.temporal_mamba = self._build_temporal_mamba()
        
        # 空间Mamba
        self.spatial_mamba = self._build_spatial_mamba()
        
        # 龙体专用注意力
        self.dragon_attention = self._build_dragon_attention()
        
        # 运动跟踪器
        self.motion_tracker = self._build_motion_tracker()
        
        # 特征融合
        self.feature_fusion = self._build_feature_fusion()
        
        # 最终解码器
        self.final_decoder = self._build_final_decoder()
    
    def _build_sam2_encoder(self):
        """构建SAM2编码器"""
        # 简化的SAM2编码器实现
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        """构建ResNet层"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_temporal_mamba(self):
        """构建时序Mamba"""
        # 简化的Mamba实现 - 减少输入维度
        class TemporalMamba(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.linear = nn.Linear(512 * 32 * 32, 1024)
                self.relu = nn.ReLU()
                self.lstm = nn.LSTM(
                    input_size=1024,
                    hidden_size=hidden_dim // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
            
            def forward(self, x):
                x = self.relu(self.linear(x))
                output, _ = self.lstm(x)
                return output
        
        return TemporalMamba(512 * 32 * 32, self.mamba_dim)
    
    def _build_spatial_mamba(self):
        """构建空间Mamba"""
        class SpatialMamba(nn.Module):
            def __init__(self, target_dim):
                super().__init__()
                self.target_dim = target_dim
                self.conv = nn.Conv2d(target_dim, target_dim, 3, padding=1)
                self.gn = nn.GroupNorm(8, target_dim)  # 使用GroupNorm替代BatchNorm
                self.relu = nn.ReLU(inplace=True)
                
            def forward(self, x):
                # 如果输入通道数不匹配，先调整
                if x.size(1) != self.target_dim:
                    x = F.adaptive_avg_pool2d(x, (32, 32))
                    x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
                    if x.size(1) != self.target_dim:
                        # 使用1x1卷积调整通道数
                        adjust_conv = nn.Conv2d(x.size(1), self.target_dim, 1).to(x.device)
                        x = adjust_conv(x)
                return self.relu(self.gn(self.conv(x)))
        
        return SpatialMamba(self.mamba_dim)
    
    def _build_dragon_attention(self):
        """构建龙体专用注意力"""
        return nn.MultiheadAttention(
            embed_dim=self.mamba_dim,
            num_heads=8,
            batch_first=True
        )
    
    def _build_motion_tracker(self):
        """构建运动跟踪器"""
        return nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # 两帧拼接
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),  # 光流
        )
    
    def _build_feature_fusion(self):
        """构建特征融合模块"""
        return nn.Sequential(
            nn.Conv2d(self.mamba_dim * 2, self.mamba_dim, 3, padding=1),
            nn.GroupNorm(8, self.mamba_dim),
            nn.ReLU(inplace=True)
        )
    
    def _build_final_decoder(self):
        """构建最终解码器"""
        return nn.Sequential(
            nn.ConvTranspose2d(self.mamba_dim, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, self.num_classes, 4, stride=2, padding=1),
        )
    
    def forward(self, images):
        """
        Args:
            images: (B, T, C, H, W) 视频帧序列
        Returns:
            segmentation_results: 分割结果字典
        """
        B, T, C, H, W = images.shape
        
        # 1. 逐帧SAM2特征提取
        frame_features = []
        for t in range(T):
            frame = images[:, t]  # (B, C, H, W)
            feat = self.sam2_encoder(frame)  # (B, 512, H/16, W/16)
            frame_features.append(feat)
        
        # 2. 时序Mamba处理
        # 将特征展平用于LSTM
        feat_flat = []
        for feat in frame_features:
            B, C, H_f, W_f = feat.shape
            feat_flat.append(feat.view(B, -1))  # (B, C*H*W)
        
        temporal_input = torch.stack(feat_flat, dim=1)  # (B, T, C*H*W)
        temporal_output = self.temporal_mamba(temporal_input)  # (B, T, mamba_dim)
        
        # 3. 处理每一帧
        frame_results = []
        for t in range(T):
            # 获取当前帧的时序特征
            temp_feat = temporal_output[:, t]  # (B, mamba_dim)
            # 计算正确的特征图尺寸
            feat_size = int((temp_feat.size(1) / self.mamba_dim) ** 0.5) if temp_feat.size(1) >= self.mamba_dim else 1
            if feat_size * feat_size * self.mamba_dim != temp_feat.size(1):
                feat_size = 1
                channels = temp_feat.size(1)
            else:
                channels = self.mamba_dim
            temp_feat = temp_feat.view(B, channels, feat_size, feat_size)  # 重塑为特征图
            
            # 空间Mamba精细化
            refined_features = self.spatial_mamba(temp_feat)  # (B, mamba_dim, 32, 32)
            
            # 龙体专用注意力
            B, C, H_r, W_r = refined_features.shape
            feat_tokens = refined_features.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            attended_feat, _ = self.dragon_attention(feat_tokens, feat_tokens, feat_tokens)
            attended_features = attended_feat.permute(0, 2, 1).view(B, C, H_r, W_r)
            
            # 运动信息（如果不是第一帧）
            if t > 0:
                prev_frame = images[:, t-1]
                curr_frame = images[:, t]
                motion_input = torch.cat([prev_frame, curr_frame], dim=1)  # (B, 6, H, W)
                motion_flow = self.motion_tracker(motion_input)
            else:
                motion_flow = None
            
            # 特征融合
            fused_input = torch.cat([refined_features, attended_features], dim=1)
            fused_features = self.feature_fusion(fused_input)
            
            # 最终分割
            segmentation_mask = self.final_decoder(fused_features)
            
            frame_results.append({
                'mask': segmentation_mask,
                'features': fused_features,
                'motion_flow': motion_flow
            })
        
        return {
            'masks': torch.stack([r['mask'] for r in frame_results], dim=1),  # (B, T, num_classes, H, W)
            'features': [r['features'] for r in frame_results],
            'motion_flows': [r['motion_flow'] for r in frame_results if r['motion_flow'] is not None]
        }