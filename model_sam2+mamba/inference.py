import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from model import DragonSegmentationHybrid
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DragonVideoProcessor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = DragonSegmentationHybrid()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 预处理
        self.transform = A.Compose([
            A.Resize(1024, 1024),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def process_video(self, video_path, output_path, sequence_length=8):
        """处理完整视频"""
        cap = cv2.VideoCapture(str(video_path))
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_buffer = []
        frame_count = 0
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_buffer.append(frame)
            frame_count += 1
            
            # 批处理模式
            if len(frame_buffer) == sequence_length:
                masks = self._process_frame_batch(frame_buffer)
                
                # 保存结果
                for i, (frame, mask) in enumerate(zip(frame_buffer, masks)):
                    result_frame = self._overlay_mask(frame, mask)
                    out.write(result_frame)
                
                # 滑窗处理
                frame_buffer = frame_buffer[sequence_length//2:]
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        # 处理剩余帧
        if frame_buffer:
            # 填充到sequence_length
            while len(frame_buffer) < sequence_length:
                frame_buffer.append(frame_buffer[-1])
            
            masks = self._process_frame_batch(frame_buffer)
            for frame, mask in zip(frame_buffer, masks):
                result_frame = self._overlay_mask(frame, mask)
                out.write(result_frame)
        
        cap.release()
        out.release()
        print(f"Video processing completed. Output saved to {output_path}")
    
    def _process_frame_batch(self, frames):
        """批处理帧序列"""
        with torch.no_grad():
            # 预处理
            processed_frames = []
            original_sizes = []
            
            for frame in frames:
                original_sizes.append(frame.shape[:2])
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                transformed = self.transform(image=frame_rgb)
                processed_frame = transformed['image']
                processed_frames.append(processed_frame)
            
            # 堆叠为批次
            batch = torch.stack(processed_frames, dim=0).unsqueeze(0)  # (1, T, C, H, W)
            batch = batch.to(self.device)
            
            # 模型推理
            predictions = self.model(batch)
            masks = predictions['masks']  # (1, T, C, H, W)
            
            # 后处理
            masks = torch.sigmoid(masks).squeeze(0)  # (T, C, H, W)
            masks = masks.cpu().numpy()
            
            # 调整回原始尺寸
            result_masks = []
            for i, (mask, original_size) in enumerate(zip(masks, original_sizes)):
                # 取第一个类别（假设是龙）
                mask = mask[1] if mask.shape[0] > 1 else mask[0]
                mask = cv2.resize(mask, (original_size[1], original_size[0]))
                mask = (mask > 0.5).astype(np.uint8)
                result_masks.append(mask)
            
            return result_masks
    
    def _overlay_mask(self, frame, mask, alpha=0.6):
        """在帧上叠加掩码"""
        # 创建彩色掩码
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = [0, 255, 0]  # 绿色
        
        # 叠加
        result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        return result

def main():
    parser = argparse.ArgumentParser(description='Dragon Video Segmentation Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--output_path', type=str, required=True, help='Output video path')
    parser.add_argument('--sequence_length', type=int, default=8, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = DragonVideoProcessor(args.model_path, args.device)
    
    # 处理视频
    processor.process_video(
        video_path=args.video_path,
        output_path=args.output_path,
        sequence_length=args.sequence_length
    )

if __name__ == '__main__':
    main()