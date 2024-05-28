import cv2
import numpy as np

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """应用CLAHE（对比度受限自适应直方图均衡化）增强对比度"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img

def color_balance(image, percent):
    """颜色平衡"""
    assert 0 < percent < 100, "percent must be in the range (0, 100)"
    half_percent = percent / 200.0

    channels = cv2.split(image)
    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        height, width = channel.shape
        vec_size = width * height
        flat = channel.flatten()
        flat = np.sort(flat)

        low_val = flat[int(vec_size * half_percent)]
        high_val = flat[int(vec_size * (1 - half_percent))]

        thresholded = np.clip(channel, low_val, high_val)
        normalized = cv2.normalize(thresholded, None, 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    balanced = cv2.merge(out_channels)
    return balanced

def dehaze(image_path, output_path):
    """去雾主函数"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像 {image_path}")
    
    # 温和增强对比度
    enhanced_image = enhance_contrast(image, clip_limit=1.0, tile_grid_size=(16, 16))
    
    # 颜色平衡
    balanced_image = color_balance(enhanced_image, 2)  # 使用1%的颜色平衡
    # balanced_image = enhanced_image

    cv2.imwrite(output_path, balanced_image)
# 使用示例
in_img = '34_outdoor_hazy_h_0_w_27.png'
dehaze(in_img, 'dehazed_image1.jpg')
