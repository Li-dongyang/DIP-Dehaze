import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取去雾后的图像
dehazed_image_path = '34_outdoor_hazy_h_0_w_27.png'
image = cv2.imread(dehazed_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 应用双边滤波
bilateral_filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# 显示原图像和处理后的图像
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(image)
ax[0].set_title('Dehazed Image')
ax[0].axis('off')

ax[1].imshow(bilateral_filtered_image)
ax[1].set_title('Bilateral Filtered Image')
ax[1].axis('off')

plt.show()
plt.savefig('bilateral_filter.png')
