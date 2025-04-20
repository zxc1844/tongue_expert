import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path so we can import the image_preprocessing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_preprocessing import (
    white_balance, 
    light_normalization, 
    color_correction, 
    retinex_enhancement,
    gamma_correction, 
    preprocess_image, 
    load_and_preprocess_image
)

# 设置默认输出目录
DEFAULT_OUTPUT_DIR = Path("data/preprocessing_results")

def display_image_comparison(original, processed, title):
    """
    Display the original and processed images side by side for comparison
    
    Args:
        original (numpy.ndarray): The original image
        processed (numpy.ndarray): The processed image
        title (str): The title for the figure
    """
    # Convert from BGR to RGB for display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('原始图像 (Original)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_rgb)
    plt.title(f'处理后 ({title})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def test_preprocessing_steps(image_path, save_dir=None):
    """
    Test each preprocessing step and the complete preprocessing pipeline
    
    Args:
        image_path (str): Path to the test image
        save_dir (str, optional): Directory to save the processed images
    """
    # 如果未指定输出目录，使用默认目录
    if save_dir is None:
        # 创建时间戳子目录，避免覆盖之前的结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = DEFAULT_OUTPUT_DIR / timestamp
    else:
        save_dir = Path(save_dir)
    
    # 创建保存目录
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"图像处理结果将保存至: {save_dir.absolute()}")
    
    # 加载原始图像
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 获取不带扩展名的基本文件名
    base_name = Path(image_path).stem
    
    # 保存原始图像副本到输出目录
    original_save_path = save_dir / f"{base_name}_original.jpg"
    cv2.imwrite(str(original_save_path), original)
    print(f"已保存原始图像: {original_save_path}")
    
    # 测试每个预处理步骤
    steps = [
        ("白平衡 (White Balance)", white_balance, "white_balance"),
        ("光照归一化 (Light Normalization)", light_normalization, "light_normalization"),
        ("色彩校正 (Color Correction)", color_correction, "color_correction"),
        ("阴影去除增强 (Retinex Enhancement)", retinex_enhancement, "retinex_enhancement"),
        ("伽马校正 (Gamma Correction)", lambda img: gamma_correction(img, gamma=1.2), "gamma_correction"),
        ("完整预处理流程 (Complete Pipeline)", preprocess_image, "complete_pipeline")
    ]
    
    # 创建一个HTML报告文件
    report_path = save_dir / "processing_report.html"
    with open(report_path, 'w', encoding='utf-8') as report:
        report.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>舌诊图像预处理结果</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .comparison {{ display: flex; margin-bottom: 30px; }}
                .image {{ margin-right: 20px; }}
                h1, h2 {{ color: #333; }}
                img {{ max-width: 500px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>舌诊图像预处理结果对比</h1>
            <p>原始图像: {image_path}</p>
            <p>处理时间: {timestamp}</p>
            
        """)
        
        # 处理每个步骤并保存结果
        for title, func, file_suffix in steps:
            print(f"应用 {title}...")
            
            # 应用预处理步骤
            processed = func(original.copy())
            
            # 显示对比图
            display_image_comparison(original, processed, title)
            
            # 保存处理后的图像，使用英文文件名
            output_path = save_dir / f"{base_name}_{file_suffix}.jpg"
            cv2.imwrite(str(output_path), processed)
            print(f"已保存 {title} 结果: {output_path}")
            
            # 将对比添加到HTML报告
            report.write(f"""
            <h2>{title}</h2>
            <div class="comparison">
                <div class="image">
                    <h3>原始图像</h3>
                    <img src="{original_save_path.name}" alt="原始图像">
                </div>
                <div class="image">
                    <h3>处理后</h3>
                    <img src="{output_path.name}" alt="{title}">
                </div>
            </div>
            """)
        
        # 完成HTML报告
        report.write("""
        </body>
        </html>
        """)
    
    print(f"\n所有处理步骤已完成。")
    print(f"可视化报告已保存至: {report_path}")
    print(f"您可以在浏览器中打开此HTML文件查看所有处理结果的对比。")

def main():
    parser = argparse.ArgumentParser(description="测试舌诊图像预处理步骤并保存对比结果")
    parser.add_argument("image_path", help="测试图像的路径")
    parser.add_argument("--save_dir", help="保存处理后图像的目录（默认：data/preprocessing_results/时间戳）", default=None)
    
    args = parser.parse_args()
    
    test_preprocessing_steps(args.image_path, args.save_dir)

if __name__ == "__main__":
    main() 