## 使用方法
### 训练模型
```
python train.py --data_dir ./dataset 
--batch_size 2 --sequence_length 8 --use_wandb
```
### 推理
```
python inference.py --model_path ./checkpoints/
best_model.pth --video_path input_video.mp4 
--output_path output_video.mp4
```
## 8. 依赖安装
```
pip install torch torchvision torchaudio
pip install transformers
pip install mamba-ssm
pip install opencv-python
pip install albumentations
pip install wandb
pip install tqdm
```