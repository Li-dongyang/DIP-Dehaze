import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.segmentation import slic
from skimage.color import rgb2lab, rgb2hsv

def cap_depth_estimation(image):
    # 将图像转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv_image[:, :, 2] / 255.0
    s = hsv_image[:, :, 1] / 255.0
    # CAP 深度估计公式
    theta0, theta1, theta2 = 0.121779, 0.959710, 0.780245
    depth_map = theta0 + theta1 * v - theta2 * s
    return depth_map

def dslic(image, num_segments=100, compactness=10):
    # 将图像转换为 CIELAB 颜色空间
    lab_image = rgb2lab(image)
    hsv_image = rgb2hsv(image)
    
    # 计算深度图
    depth_map = cap_depth_estimation(image)
    
    # 构建 7 维特征向量
    height, width, _ = image.shape
    feature_vector = np.zeros((height, width, 7))
    feature_vector[:, :, :3] = lab_image
    feature_vector[:, :, 3:5] = np.indices((height, width)).transpose(1, 2, 0) # (height, width, 2)
    feature_vector[:, :, 5] = hsv_image[:, :, 1]
    feature_vector[:, :, 6] = depth_map
    
    # 执行 SLIC 超像素分割
    segments = slic(feature_vector, n_segments=num_segments, compactness=compactness, start_label=1)
    
    return segments

# 读取输入图像
input_image_path = '34_outdoor_hazy_h_0_w_27.png'
image = cv2.imread(input_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 开始计时
start_time = time.time()

# 执行 DSLIC 超像素分割
segments = slic(image, n_segments=200, compactness=10, channel_axis=2)

# 结束计时
end_time = time.time()
elapsed_time = end_time - start_time

# 显示结果
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(image)
ax[0].set_title('Input Image')
ax[0].axis('off')

ax[1].imshow(segments)
ax[1].set_title(f'DSLIC Superpixels\nElapsed Time: {elapsed_time:.2f} seconds')
ax[1].axis('off')

# plt.show()
plt.savefig('dslc_superpixels.png')
