import os
import json
from labelme2coco import convert

def convert_labelme_to_coco():
    # 设置路径
    labelme_folder = "/Users/ericw/project/sitp_wulong/dataset/images"
    save_json_path = "/Users/ericw/project/sitp_wulong/dataset/coco_form/annotations.json"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(save_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 检查输入文件夹是否存在
    if not os.path.exists(labelme_folder):
        print(f"错误：输入文件夹不存在 {labelme_folder}")
        return
    
    # 检查是否有JSON标注文件
    json_files = [f for f in os.listdir(labelme_folder) if f.endswith('.json')]
    if not json_files:
        print("错误：在指定文件夹中没有找到JSON标注文件")
        return
    
    print(f"找到 {len(json_files)} 个标注文件")
    
    try:
        # 执行转换
        convert(labelme_folder, save_json_path)
        print(f"转换成功！COCO格式文件已保存到: {save_json_path}")
    except Exception as e:
        print(f"转换失败：{e}")

if __name__ == "__main__":
    convert_labelme_to_coco()