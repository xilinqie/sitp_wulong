# 使用方法：
# # 查看可用的视频文件
# python video_frame_extractor.py
# # 处理单个视频文件
# python video_frame_extractor.py -v your_video.mp4
# # 处理所有视频文件
# python video_frame_extractor.py --all
# # 自定义间隔时间
# python video_frame_extractor.py --all -i 5



import os
import subprocess
import argparse
import sys

class VideoFrameExtractor:
    def __init__(self, output_dir="/Users/ericw/project/sitp_wulong/dataset/images"):
        self.output_dir = output_dir
        self.input_dir = "/Users/ericw/project/sitp_wulong/dataset/video"
        self.ensure_output_dir()
        self.ensure_input_dir()
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
    
    def ensure_input_dir(self):
        if not os.path.exists(self.input_dir):
            print(f"警告: 输入目录不存在 - {self.input_dir}")
            return False
        return True
    
    def check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_video_files(self):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        for file in os.listdir(self.input_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(self.input_dir, file))
        
        return sorted(video_files)
    
    def extract_frames(self, video_path, interval=3):
        # Args:
        #     video_path (str): 输入视频文件路径
        #     interval (int): 提取间隔秒数，默认3秒
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在 - {video_path}")
            return False
        
        if not self.check_ffmpeg():
            print("错误: 未找到ffmpeg，请确保已安装ffmpeg")
            return False
        
        # 获取视频文件名（不含扩展名）作为前缀
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_pattern = os.path.join(self.output_dir, f"{video_name}_img%04d.jpg")
        
        cmd = [
            'ffmpeg',
            '-i', video_path,           # 输入视频文件
            '-vf', f'fps=1/{interval}', # 视频滤镜：每interval秒提取一帧
            '-y',                       # 覆盖输出文件
            output_pattern              # 输出文件名模式
        ]
        
        try:
            print(f"开始提取帧: {video_name}，每{interval}秒一帧...")
            print(f"输出目录: {self.output_dir}")
            
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            
            # 统计生成的图片数量
            jpg_files = [f for f in os.listdir(self.output_dir) 
                        if f.startswith(video_name) and f.endswith('.jpg')]
            print(f"成功提取 {len(jpg_files)} 帧图片")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg执行失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            return False
    
    def process_all_videos(self, interval=3):
        video_files = self.get_video_files()
        
        if not video_files:
            print(f"在目录 {self.input_dir} 中未找到视频文件")
            return False
        
        print(f"找到 {len(video_files)} 个视频文件:")
        for video in video_files:
            print(f"  - {os.path.basename(video)}")
        
        success_count = 0
        for video_path in video_files:
            print(f"\n处理视频: {os.path.basename(video_path)}")
            if self.extract_frames(video_path, interval):
                success_count += 1
            else:
                print(f"处理失败: {os.path.basename(video_path)}")
        
        print(f"\n处理完成: {success_count}/{len(video_files)} 个视频成功处理")
        return success_count == len(video_files)

def main():
    parser = argparse.ArgumentParser(description='从指定目录的视频中每隔指定秒数提取帧')
    parser.add_argument('-i', '--interval', type=int, default=3, 
                       help='提取间隔秒数 (默认: 3秒)')
    parser.add_argument('-v', '--video', type=str, 
                       help='指定单个视频文件名（在输入目录中）')
    parser.add_argument('--all', action='store_true',
                       help='处理输入目录中的所有视频文件')
    
    args = parser.parse_args()
    
    # 创建提取器实例
    extractor = VideoFrameExtractor()
    
    if args.video:
        # 处理单个视频文件
        video_path = os.path.join(extractor.input_dir, args.video)
        success = extractor.extract_frames(video_path, args.interval)
        
        if success:
            print("帧提取完成!")
        else:
            print("帧提取失败!")
            sys.exit(1)
    
    elif args.all:
        # 处理所有视频文件
        success = extractor.process_all_videos(args.interval)
        
        if success:
            print("所有视频帧提取完成!")
        else:
            print("部分或全部视频帧提取失败!")
            sys.exit(1)
    
    else:
        print("请指定要处理的视频:")
        print("  使用 -v filename.mp4 处理单个视频")
        print("  使用 --all 处理所有视频")
        print("\n可用的视频文件:")
        video_files = extractor.get_video_files()
        if video_files:
            for video in video_files:
                print(f"  - {os.path.basename(video)}")
        else:
            print("  (未找到视频文件)")

if __name__ == "__main__":
    main()