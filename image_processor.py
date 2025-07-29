import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import time

class ImageProcessor:
    def __init__(self):
        pass
    
    def analyze_brightness(self, image):
        """
        分析图像的亮度
        返回一个介于0（暗）和1（亮）之间的值
        """
        # 对于彩色图像，转换为HSV并使用V通道
        if len(image.shape) == 3:
            # 将BGR转换为HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # 提取V通道（亮度）
            v_channel = hsv[:, :, 2]
            # 计算平均亮度（归一化为0-1）
            return np.mean(v_channel) / 255.0
        else:
            # 对于灰度图像，直接计算平均亮度
            return np.mean(image) / 255.0
    
    def get_adaptive_brightness_adjustment(self, brightness_level):
        """
        根据当前图像亮度确定亮度调整
        Args:
            brightness_level：介于0和1之间的值，表示当前亮度
        Returns:
            应用百分比调整
        """
        # 暗图像（亮度级别<0.3）：应用大调整（+100至+120）
        if brightness_level < 0.3:
            # 图像越暗，调整越强
            return 50 + (0.3 - brightness_level) * 200
            
        # 中等亮度（0.3<=brightness_level<0.6）：进行适度调整
        elif brightness_level < 0.6:
            return 15 + (0.6 - brightness_level) * 80
            
        # 明亮的图像（brightness_level>=0.6）：应用最小调整
        else:
            # 对于非常明亮的图像，可能需要进行轻微的负向调整
            return 5 - (brightness_level - 0.6) * 40
    
    def adjust_brightness(self, image, percentage):
        """按百分比调整亮度（-100到100之间）"""
        factor = 1 + percentage / 100
        return np.clip(image * factor, 0, 255).astype(np.uint8)
        
    def adjust_contrast(self, image, percentage):
        """
        使用基于曲线的方法按百分比（-100到100）调整对比度
        方法提供了类似于照片编辑软件的更自然的对比度调整
        """
        # 将图像百分比转换为合理的系数（0到3）
        # 将图像百分比范围-100到100映射到因子范围0到3
        factor = max(0, (percentage + 100) / 100)
        
        # 转换为浮点数进行处理
        img_float = image.astype(np.float32) / 255.0
        
        # 应用对比度曲线（使用类似S形的函数）
        # 创建一条S曲线，在阴影和高光中保留更多细节
        if percentage > 0:
            # 正对比：应用S曲线
            img_float = (1.0 / (1.0 + np.exp(-(img_float - 0.5) * factor * 3))) 
            # 乘以 3 可以调整曲线的陡峭程度，让这种效果在较低百分比时更明显。
        else:
            # 负对比度：应用反向 S 曲线
            img_float = 0.5 + (img_float - 0.5) * factor
        
        # 像素值转换回 0-255 范围
        return np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    def enhance_shadows(self, image, percentage):
        """
        按百分比（0-100）增强阴影亮度
        此操作可在保留高光区域的同时提高暗部区域的亮度
        """
        # 转换为浮点数进行处理
        img_float = image.astype(np.float32)
        
        # 定义阴影阈值（低于此值的像素被视为阴影）
        threshold = 100
        
        # 创建权重掩码，对于完整阴影为1，对于明亮区域为0
        shadow_mask = np.clip((threshold - img_float) / threshold, 0, 1)
        
        # 计算增亮量（50% 意味着在最暗区域最多添加最大值的 50%）
        brightening = shadow_mask * (255 * percentage / 100)
        
        # 对图像应用增亮处理
        result = np.clip(img_float + brightening, 0, 255).astype(np.uint8)
        return result
    
    def enhance_structure(self, image, percentage):
        """Enhance structure by percentage (0 to 100+)"""
        # 转换为浮点数进行处理
        img_float = image.astype(np.float32) / 255.0
        
        # 使用双边滤波器提取结构层
        sigma_color, sigma_space = 10, 10
        structure = cv2.bilateralFilter(img_float, -1, sigma_color, sigma_space)
        
        # 提取细节层
        detail = img_float - structure
        
        # 根据百分比增强细节层
        strength = 1 + percentage / 100
        enhanced = structure + detail * strength
        
        # 裁剪数值并转换回原格式
        result = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        return result
    
    def process_image(self, image, brightness=None, contrast=30, shadows=50, structure=100):
        """
        使用可配置参数对图像应用所有调整
        """
        # 制作一个副本，以避免修改原始内容
        result = image.copy()
        
        # 分析图像亮度，若未指定调整参数则确定调整方案
        if brightness is None:
            img_brightness = self.analyze_brightness(image)
            brightness = self.get_adaptive_brightness_adjustment(img_brightness)
            print(f"图像亮度等级: {img_brightness:.2f}, 应用调整: {brightness:.1f}%")
        
        # 对于彩色图像，处理每个通道
        if len(image.shape) == 3:
            for i in range(3):  # 处理每个颜色通道（多通道图像）
                # 按顺序应用调整
                result[:,:,i] = self.enhance_shadows(result[:,:,i], shadows)
                
                result[:,:,i] = self.adjust_brightness(result[:,:,i], brightness)
                
                result[:,:,i] = self.adjust_contrast(result[:,:,i], contrast)
                
                result[:,:,i] = self.enhance_structure(result[:,:,i], structure)
        else:
            # 灰度图像处理（单通道图像）
            result = self.adjust_brightness(result, brightness)
            result = self.adjust_contrast(result, contrast)
            result = self.enhance_shadows(result, shadows)
            result = self.enhance_structure(result, structure)
            
        return result

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Process images with specified adjustments')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory to save processed images')
    parser.add_argument('--brightness', type=float, default=None, help='Brightness adjustment (-100 to 100) or None for automatic')
    parser.add_argument('--contrast', type=float, default=95, help='Contrast adjustment (-100 to 100)')
    parser.add_argument('--shadows', type=float, default=20, help='Shadow enhancement (0 to 100)')
    parser.add_argument('--structure', type=float, default=25, help='Structure enhancement (0 to 100+)')
    args = parser.parse_args()
    
    # 如果输出目录不存在，则创建它
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化图像处理
    processor = ImageProcessor()
    
    # 图像支持格式设置
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 处理输入文件夹中的每一张图象
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    
    print(f"输入文件下发现 {len(image_files)} 张图像需要处理.")
    
    for img_path in image_files:
        try:
            print(f"处理图像： {img_path.name}...")
            # 开始计时
            start_time = time.time()
            
            # 读取图像
            image = cv2.imread(str(img_path))
            
            if image is None:
                print(f"读取图像错误 {img_path.name}, 跳过...提示：请重新对该图像进行命名...")
                continue
                
            # 处理过程
            processed = processor.process_image(
                image, 
                brightness=args.brightness,
                contrast=args.contrast,
                shadows=args.shadows,
                structure=args.structure
            )
            # 计算每张图处理时间（秒）以及FPS
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            print(f"处理时间: {elapsed_time:.4f} 秒")
            print(f"FPS: {fps:.2f}")
            # 保存图像
            output_file = output_path / img_path.name
            cv2.imwrite(str(output_file), processed)
            
            print(f"已保存处理后的图像至 {output_file.name}")
            
        except Exception as e:
            print(f"存在图像处理错误，发生于图像 {img_path.name}: {str(e)}")
    
    print("处理完成!")

if __name__ == "__main__":
    main()