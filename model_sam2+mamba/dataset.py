import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DragonVideoDataset(Dataset):
    """舞龙视频数据集"""
    
    def __init__(self, 
                 data_dir,
                 sequence_length=8,
                 image_size=(1024, 1024),
                 is_training=True,
                 overlap_ratio=0.5):
        
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.is_training = is_training
        self.overlap_ratio = overlap_ratio
        
        # 加载数据
        self.sequences = self._load_sequences()
        
        # 数据增强
        self.transform = self._get_transforms()
        
    def _load_sequences(self):
        """加载视频序列"""
        sequences = []
        
        # 数据组织为: images/frame_xxx.jpg 和对应的 frame_xxx.json
        image_dir = self.data_dir / 'images'
        
        # 获取所有图像文件
        image_files = []
        json_files = []
        
        for img_file in sorted(image_dir.glob('*.jpg')):
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                image_files.append(img_file)
                json_files.append(json_file)
        
        print(f"Found {len(image_files)} image-annotation pairs")
        
        # 创建序列 - 滑动窗口方式
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        for i in range(0, len(image_files) - self.sequence_length + 1, step_size):
            sequence_images = image_files[i:i + self.sequence_length]
            sequence_annotations = json_files[i:i + self.sequence_length]
            
            sequences.append({
                'images': sequence_images,
                'annotations': sequence_annotations
            })
        
        print(f"Created {len(sequences)} sequences with length {self.sequence_length}")
        return sequences
    
    def _get_transforms(self):
        """获取数据增强变换"""
        if self.is_training:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 加载图像和掩码
        images = []
        masks = []
        
        for img_path, ann_path in zip(sequence['images'], sequence['annotations']):
            # 加载图像
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 加载标注
            with open(ann_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # 生成掩码
            mask = self._create_mask_from_annotation(annotation, image.shape[:2])
            
            # 应用变换
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            images.append(image)
            masks.append(mask)
        
        # 堆叠为张量
        images = torch.stack(images, dim=0)  # (T, C, H, W)
        masks = torch.stack(masks, dim=0)    # (T, H, W)
        
        return {
            'images': images,
            'masks': masks,
            'sequence_id': idx
        }
    
    def _create_mask_from_annotation(self, annotation, image_shape):
        """从标注创建掩码"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # 处理LabelMe格式的标注
        if 'shapes' in annotation:
            for shape in annotation['shapes']:
                if shape['label'] == 'dragon':  # 假设龙的标签是'dragon'
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)
        
        return mask

def create_data_loaders(data_dir, batch_size=2, num_workers=4, sequence_length=8):
    """创建数据加载器"""
    
    # 训练集
    train_dataset = DragonVideoDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        is_training=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 验证集（使用相同数据，实际应该分开）
    val_dataset = DragonVideoDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        is_training=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader