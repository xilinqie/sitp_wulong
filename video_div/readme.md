# Video Frame Extractor

使用ffmpeg从视频中每隔指定秒数提取帧图片的Python工具。

## 功能特性

- 每隔指定秒数（默认3秒）提取一帧
- 输出为JPG格式图片
- 自动命名为img0001.jpg, img0002.jpg等格式
- 支持自定义输出目录和提取间隔

## 安装要求

### 1. 安装ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
下载ffmpeg并添加到系统PATH

### 2. Python环境
- Python 3.6+

## 使用方法

### 基本用法
```bash
python video_frame_extractor.py your_video.mp4
```

### 自定义参数
```bash
# 每5秒提取一帧
python video_frame_extractor.py your_video.mp4 -i 5

# 指定输出目录
python video_frame_extractor.py your_video.mp4 -o my_frames

# 组合使用
python video_frame_extractor.py your_video.mp4 -i 2 -o custom_output
```

### 参数说明
- `video_path`: 输入视频文件路径（必需）
- `-i, --interval`: 提取间隔秒数（默认3秒）
- `-o, --output`: 输出目录（默认output_frames）

## 输出格式

提取的图片将保存在指定目录中，命名格式为：
- img0001.jpg
- img0002.jpg
- img0003.jpg
- ...

## 示例

```bash
# 从sample.mp4每3秒提取一帧到output_frames目录
python video_frame_extractor.py sample.mp4

# 每10秒提取一帧到frames目录
python video_frame_extractor.py sample.mp4 -i 10 -o frames


# 基本用法（每3秒提取一帧）
python video_frame_extractor.py your_video.mp4

# 自定义间隔（每5秒提取一帧）
python video_frame_extractor.py your_video.mp4 -i 5

new
# 查看可用的视频文件
python video_frame_extractor.py

# 处理单个视频文件
python video_frame_extractor.py -v your_video.mp4

# 处理所有视频文件
python video_frame_extractor.py --all

# 自定义间隔时间
python video_frame_extractor.py --all -i 5
```