import os
from PIL import Image

from config import HAZERD_ROOT
from math import ceil
from tqdm import tqdm

# MODE="val"
MODE="test"

if __name__ == '__main__':
    hazerd_root = HAZERD_ROOT
    crop_size = 256
    result_foldername = f"{MODE}_crop_{str(crop_size)}"

    ori_root = os.path.join(hazerd_root, r'data')
    ori_haze_root = os.path.join(ori_root, 'simu')
    ori_gt_root = os.path.join(ori_root, 'img')

    patch_root = os.path.join(hazerd_root, result_foldername)
    patch_haze_path = os.path.join(patch_root, 'hazy')
    patch_gt_path = os.path.join(patch_root, 'gt')

    os.makedirs(patch_root, exist_ok=True)
    os.makedirs(patch_haze_path, exist_ok=True)
    os.makedirs(patch_gt_path, exist_ok=True)

    # 35-40 for val and 41+ for test
    train_list = [img_name for img_name in os.listdir(ori_haze_root)]

    for idx, img_name in enumerate(tqdm(train_list)):

        img = Image.open(os.path.join(ori_haze_root, img_name))
        gt_img_name = img_name.rsplit('_', 1)[0] + '_RGB.jpg'
        gt = Image.open(os.path.join(ori_gt_root, gt_img_name))

        assert img.size == gt.size

        w, h = img.size
        stride = int(crop_size / 3.)
        h_steps = 1 + int(ceil(float(max(h - crop_size, 0)) / stride))
        w_steps = 1 + int(ceil(float(max(w - crop_size, 0)) / stride))

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                ws0 = w_idx * stride
                ws1 = crop_size + ws0
                hs0 = h_idx * stride
                hs1 = crop_size + hs0
                if h_idx == h_steps - 1:
                    hs0, hs1 = max(h - crop_size, 0), h
                if w_idx == w_steps - 1:
                    ws0, ws1 = max(w - crop_size, 0), w
                img_crop = img.crop((ws0, hs0, ws1, hs1))
                gt_crop = gt.crop((ws0, hs0, ws1, hs1))

                img_crop.save(os.path.join(patch_haze_path, '{}_h_{}_w_{}.png'.format(idx, h_idx, w_idx)))
                gt_crop.save(os.path.join(patch_gt_path, '{}_h_{}_w_{}.png'.format(idx, h_idx, w_idx)))
