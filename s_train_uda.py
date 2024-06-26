# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from swin import dehazeformer_u, single, dehazeformer_t
from tools.config import OHAZE_ROOT
from ohaze_datasets import OHazeDataset
from tools.utils import AvgMeter, check_mkdir, sliding_forward

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='256-1000-U2',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'use_clahe': True,
    'epochs': 1000,
    'train_batch_size': 48,
    'last_iter': 0,
    'lr': 8e-4,
    'lr_decay': 0.9,
    'weight_decay': 2e-5,
    'momentum': 0.9,
    'snapshot': '',
    'val_freq': 2,
    'crop_size': 256,
}


def main():
    net = dehazeformer_u().cuda().train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfgs['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfgs['epochs'], eta_min=cfgs['lr'] * 1e-2)
	
    # if len(cfgs['snapshot']) > 0:
    #     print('training resumes from \'%s\'' % cfgs['snapshot'])
    #     net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
    #                                                 args.exp_name, cfgs['snapshot'] + '.pth')))
    #     optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
    #                                                       args.exp_name, cfgs['snapshot'] + '_optim.pth')))
    #     optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
    #     optimizer.param_groups[1]['lr'] = cfgs['lr']


    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, optimizer, scheduler)


def train(net, optimizer, scheduler):
    curr_iter = cfgs['last_iter']
    scaler = amp.GradScaler()
    torch.cuda.empty_cache()

    while curr_iter <= cfgs['epochs']:
        curr_iter += 1
        train_loss_record = AvgMeter()

        for data in tqdm(train_loader):
            haze, gt, _ = data

            batch_size = haze.size(0)

            haze, gt = haze.cuda(), gt.cuda()

            optimizer.zero_grad()

            with amp.autocast():
                output = net(haze)
                loss = criterion(output, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_record.update(loss.item(), batch_size)

            log = '[iter %d], [train loss %.5f]' % \
                  (curr_iter, train_loss_record.avg)
            # print(log)
            open(log_path, 'a').write(log + '\n')

        scheduler.step()
        if curr_iter == 1 or (curr_iter) % cfgs['val_freq'] == 0:
        # if (curr_iter + 1) % cfgs['val_freq'] == 0:
            validate(net, curr_iter, optimizer)
            torch.cuda.empty_cache()
        
def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()
    psnr_record, ssim_record = AvgMeter(), AvgMeter()

    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data
            haze, gt = haze.cuda(), gt.cuda()

            dehaze = net(haze)

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))

            for i in range(len(haze)):
                r = dehaze[i].cpu().numpy().transpose([1, 2, 0])  # data range [0, 1], channel_axis=2
                g = gt[i].cpu().numpy().transpose([1, 2, 0])
                psnr = peak_signal_noise_ratio(g, r)
                ssim = structural_similarity(g, r, data_range=1, multichannel=True, channel_axis=2,
                                             gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                psnr_record.update(psnr)
                ssim_record.update(ssim)

    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f' % (curr_iter, loss_record.avg, optimizer.param_groups[0]['lr'])
    log = '[validate]: [iter {}], [loss {:.5f}] [PSNR {:.4f}] [SSIM {:.4f}]'.format(
        curr_iter, loss_record.avg, psnr_record.avg, ssim_record.avg)
    print(log)
    open(log_path, 'a').write(log + '\n')
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(int(args.gpus))

    train_dataset = OHazeDataset(OHAZE_ROOT, f'train_crop_{cfgs["crop_size"]}', use_clahe=cfgs['use_clahe'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=36,
                              shuffle=True, drop_last=True)

    val_dataset = OHazeDataset(OHAZE_ROOT, f'val_crop_{cfgs["crop_size"]}', use_clahe=cfgs['use_clahe'])
    val_loader = DataLoader(val_dataset, batch_size=192, num_workers=36, shuffle=False, drop_last=False)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')

    main()
