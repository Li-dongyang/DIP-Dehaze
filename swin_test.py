# coding: utf-8
# CUDA_VISIBLE_DEVICES=2
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import HAZERD_ROOT, OHAZE_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from swin import dehazeformer_b, dehazeformer_u, single
from ohaze_datasets import HazeRDDataset, OHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

torch.manual_seed(100)
# torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'O-Haze-Swin-R0'
use_clahe = True
# exp_name = 'O-Haze-Swin-T1'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_2000_loss_0.05532_lr_0.000164',
    # 'snapshot': 'dehazeformer-b',
    'snapshot': '',
}

to_test = {
    # 'hazerd': HAZERD_ROOT,
    'O-Haze': OHAZE_ROOT, 
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'O-Haze' in name:
                net = dehazeformer_u().cuda()
                dataset = OHazeDataset(root, 'test_crop_256', use_clahe=use_clahe)
            elif 'hazerd' in name:
                net = dehazeformer_b().cuda()
                dataset = HazeRDDataset(root, 'test_crop_512', use_clahe=use_clahe)
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(single(
                    os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')
                ))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=240, num_workers=32, shuffle=False)

            psnrs, ssims = [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0]) # channel is the last dimension
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True, channel_axis=2,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    ssims.append(ssim)
                    # print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}'
                    #       .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim))

                # for r, f in zip(res.cpu(), fs):
                #     to_pil(r).save(
                #         os.path.join(ckpt_path, exp_name,
                #                      '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))
                print('[iter %d in %d]' % (idx + 1, len(dataloader)))
            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}")


if __name__ == '__main__':
    main()
