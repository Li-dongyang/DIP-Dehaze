import torch
from collections import OrderedDict
import cv2
import numpy as np

def single(save_dir):
	try:
		state_dict = torch.load(save_dir)['state_dict']
		new_state_dict = OrderedDict()

		for k, v in state_dict.items():
			name = k[7:]
			new_state_dict[name] = v
	except KeyError:
		new_state_dict = torch.load(save_dir)

	return new_state_dict

class CLAHE:
    def __init__(self, clip_limit=1.0, tile_grid_size=(16, 16), balance_percent=2.0):
        """初始化参数"""
        assert 0 < balance_percent < 100, "percent must be in the range (0, 100)"
        self.half_percent = balance_percent / 200.0
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image):
        """去雾主函数"""
        image = np.array(image) # channel is the last dimension
        # 温和增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # 颜色平衡
        half_percent = self.half_percent

        channels = cv2.split(final_img)
        out_channels = []
        for channel in channels:
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
        
        return balanced # channel is the last dimension