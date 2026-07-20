import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time, argparse, os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from model import PyramidFlow
from util import MOAIDataloader, BatchDiffLoss
from util import fix_randseed, compute_pro_score_fast, getLogger
from continuity import diff_loss_2d

import csv
from pathfilemgr import MPathFileManager
from hyp_data import MHyp
from utiles import remaining_time

import argparse
parser = argparse.ArgumentParser()

##########################################################################################
# path
parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')
##########################################################################################
# dataset config
parser.add_argument('--dataset', help='dataset folder name under ./dataset, e.g. mvtec_anomaly_detection')
parser.add_argument('--category', help='category folder name inside the dataset, e.g. bottle')
parser.add_argument('--dataset-root', help='local dataset root directory', default=r'E:\0_dataset')
##########################################################################################
# train config
parser.add_argument('--epoches', type=int, help='number of training epoches', default=50)
parser.add_argument('--batchSize', '--batchsize', dest='batchSize', type=int, help='training batch size', default=4)
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
##########################################################################################
# loss config. if '--loss 1', use diff_loss_2d() in continuity.py
parser.add_argument('--loss', type=int, choices=[0, 1], default=0, help='0: original Fourier loss, 1: Fourier + continuity loss')
parser.add_argument('--alpha', type=float, default=0.25, help='continuity 1st-order (smoothness) coefficient when --loss 1')
parser.add_argument('--beta', type=float, default=0.5, help='continuity 2nd-order (curvature) coefficient when --loss 1')
##########################################################################################

opt = parser.parse_args()

##########################################################################################
### make dir and hyp
##########################################################################################

base_mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)
mhyp = MHyp()
if os.path.exists(base_mpfm.hyp_yaml):
    base_mpfm.load_train_hyp(mhyp)


def prepare_run(version):
    run_mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, version)
    run_hyp = MHyp()
    if os.path.exists(run_mpfm.hyp_yaml):
        run_mpfm.load_train_hyp(run_hyp)
    if opt.epoches is not None:
        if opt.epoches <= 0:
            raise ValueError("--epoches must be greater than 0")
        run_hyp.epoches = opt.epoches
    if opt.batchSize is not None:
        if opt.batchSize <= 1:
            raise ValueError("--batchSize must be greater than 1")
        run_hyp.batchSize = opt.batchSize
    run_hyp.lossMode = opt.loss
    if opt.alpha is not None:
        if opt.alpha < 0:
            raise ValueError("--alpha must be greater than or equal to 0")
        run_hyp.continuityAlpha = opt.alpha
    if opt.beta is not None:
        if opt.beta < 0:
            raise ValueError("--beta must be greater than or equal to 0")
        run_hyp.continuityBeta = opt.beta
    run_mpfm.save_hyp(run_hyp)

    # record CLI arguments used for this run
    with open(os.path.join(run_mpfm.version_path, 'parser_argment.txt'), 'w', encoding='utf-8') as f:
        for key, value in vars(opt).items():
            f.write(f"{key}: {value}\n")

    result_file = open(run_mpfm.result_csv, mode='a', newline='', encoding='utf-8')
    result_csv = csv.writer(result_file)
    result_csv.writerow([
        "epoch",
        "time",
        "pixel_auroc",
        "image_auroc",
        "pixel_pro",
        "loss",
        "base_loss",
        "continuity_loss",
        "continuity_alpha",
        "continuity_beta",
        "loss_mode",
        "pixel_threshold",
        "image_threshold",
    ])
    twriter = SummaryWriter(run_mpfm.train_result)
    return run_mpfm, run_hyp, result_file, result_csv, twriter


def is_category_root(path):
    return (
        os.path.isdir(os.path.join(path, 'train', 'good')) or
        os.path.isdir(os.path.join(path, 'good'))
    )


def list_dataset_categories(dataset_path):
    if is_category_root(dataset_path):
        return []
    categories = []
    for name in sorted(os.listdir(dataset_path)):
        path = os.path.join(dataset_path, name)
        if os.path.isdir(path) and is_category_root(path):
            categories.append(name)
    return categories


def resolve_dataset_jobs(dataset_name, category_name, dataset_root):
    if not dataset_name:
        return [(base_mpfm.train_path, base_mpfm.val_path, 'moai', opt.version)]

    root = dataset_root if os.path.isabs(dataset_root) else os.path.join(os.getcwd(), dataset_root)
    dataset_path = os.path.join(root, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    if is_category_root(dataset_path):
        return [(dataset_path, dataset_path, dataset_name.replace('\\', '/'), opt.version)]

    categories = list_dataset_categories(dataset_path)
    if not categories:
        raise FileNotFoundError(
            f"No train/good category folders found under dataset folder: {dataset_path}"
        )

    if category_name is not None and category_name.lower() == 'all':
        return [
            (
                os.path.join(dataset_path, category),
                os.path.join(dataset_path, category),
                f'{dataset_name}/{category}',
                f'{opt.version}/{category}',
            )
            for category in categories
        ]

    if not category_name:
        raise ValueError(
            f"Dataset '{dataset_name}' contains multiple categories. "
            f"Pass --category with one of: {', '.join(categories)}"
        )

    if category_name not in categories:
        raise ValueError(
            f"Unknown category '{category_name}' for dataset '{dataset_name}'. "
            f"Available categories: {', '.join(categories)}"
        )

    category_path = os.path.join(dataset_path, category_name)
    return [(category_path, category_path, f'{dataset_name}/{category_name}', opt.version)]


def train(train_path, val_path, resnetX, num_layer, vn_dims, ksize, channel, num_stack, device, batch_size, epoches, hyp,
          run_mpfm, result_file, result_csv, twriter, cls_name='moai'):
    message = {
        "status": "prepare",
        "message": "학습 환경을 준비중입니다.",
    }

    print(json.dumps(message, ensure_ascii=False), flush=True)

    # save config
    save_dict = {'cls_name': cls_name, 'resnetX': resnetX, 'num_layer': num_layer, 'vn_dims': vn_dims,\
                 'ksize': ksize, 'channel': channel, 'num_stack': num_stack, 'batch_size': batch_size}
    loader_dict = fix_randseed(seed=opt.seed)

    # model
    flow = PyramidFlow(resnetX, channel, num_layer, num_stack, ksize, vn_dims, False).to(device)
    x_size = 256 if resnetX==0 else 1024
    # optimizer = torch.optim.Adam(flow.parameters(), lr=2e-4, eps=1e-04, weight_decay=1e-5, betas=(0.5, 0.9)) # using cs-flow optimizer
    optimizer = torch.optim.AdamW(flow.parameters(), lr=2e-4, eps=1e-04, weight_decay=1e-5, betas=(0.5, 0.9))
    Loss = BatchDiffLoss(batch_size, p=2)
    loss_mode = int(getattr(hyp, "lossMode", 0))
    continuity_alpha = float(getattr(hyp, "continuityAlpha", 0.25))
    continuity_beta = float(getattr(hyp, "continuityBeta", 0.5))
    continuity_mode = getattr(hyp, "continuityMode", "l1")
    pixel_threshold_percentile = float(getattr(hyp, "pixelThresholdPercentile", 99.5))
    image_threshold_percentile = float(getattr(hyp, "imageThresholdPercentile", 95.0))
    save_dict['loss_mode'] = loss_mode
    save_dict['continuity_alpha'] = continuity_alpha if loss_mode == 1 else 0.0
    save_dict['continuity_beta'] = continuity_beta if loss_mode == 1 else 0.0

    # dataset
    train_dataset = MOAIDataloader(mode='train', x_size=x_size, y_size=256, datapath=train_path, hyp=hyp)
    val_dataset = MOAIDataloader(mode='val', x_size=x_size, y_size=256, datapath=train_path, hyp=hyp)
    test_dataset = MOAIDataloader(mode='test', x_size=x_size, y_size=256, datapath=val_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=True, drop_last=True, **loader_dict)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, drop_last=False, **loader_dict)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, **loader_dict)

    # training & evaluation
    pixel_auroc_lst = [0]
    pixel_pro_lst = [0]
    image_auroc_lst = [0]
    losses_lst = [0]
    t0 = time.time()

    message = {
        "status": "start",
        "message": "모델 학습을 시작합니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

    for epoch in range(epoches):
        # train
        flow.train()
        losses = []
        base_losses = []
        continuity_losses = []
        for train_dict in train_loader:
            image, labels = train_dict['images'].to(device), train_dict['labels'].to(device)
            optimizer.zero_grad()
            pyramid2= flow(image)
            diffes = Loss(pyramid2)
            diff_pixel = flow.pyramid.compose_pyramid(diffes).mean(1)  
            base_loss = torch.fft.fft2(diff_pixel).abs().mean() # Fourier loss
            if loss_mode == 1:
                cont_loss = pyramid_continuity_loss(
                    pyramid2,
                    alpha=continuity_alpha,
                    beta=continuity_beta,
                    mode=continuity_mode,
                )
                loss = base_loss + cont_loss
            else:
                cont_loss = base_loss.new_tensor(0.0)
                loss = base_loss
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1e0) # Avoiding numerical explosions
            optimizer.step()
            losses.append(loss.item())
            base_losses.append(base_loss.item())
            continuity_losses.append(cont_loss.item())
        mean_loss = float(np.mean(losses))
        mean_base_loss = float(np.mean(base_losses))
        mean_continuity_loss = float(np.mean(continuity_losses))
        losses_lst.append(mean_loss)

        # val for template 
        flow.eval()
        feat_sum, cnt = [0 for _ in range(num_layer)], 0
        for val_dict in val_loader:
            image = val_dict['images'].to(device)
            with torch.no_grad():
                pyramid2= flow(image) 
                cnt += 1
            feat_sum = [p0+p for p0, p in zip(feat_sum, pyramid2)]
        feat_mean = [p/cnt for p in feat_sum]

        savefm(feat_mean, run_mpfm)
        pixel_threshold, image_threshold = estimate_thresholds(
            flow,
            val_loader,
            feat_mean,
            device,
            pixel_percentile=pixel_threshold_percentile,
            image_percentile=image_threshold_percentile,
        )

        # test
        flow.eval()
        diff_list, labels_list = [], []
        for test_dict in test_loader:
            image, labels, fname = test_dict['images'].to(device), test_dict['labels'], test_dict['fname']
            with torch.no_grad():
                pyramid2 = flow(image) 
                pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid2, feat_mean)]
                diff = flow.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)# b,1,h,w
                diff_list.append(diff.cpu())
                labels_list.append(labels.cpu()==1)# b,1,h,w

                diff = np.squeeze(diff.cpu().numpy())
                amap_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                np.savez(f'{run_mpfm.train_result}/{fname[0]}.npz', amap_norm=amap_norm)
                # plt.imsave(f'{run_mpfm.train_result}/{fname[0]}.png', amap_norm, cmap='jet')

        labels_all = torch.concat(labels_list, dim=0)# b1hw 
        amaps = torch.concat(diff_list, dim=0)# b1hw 
        amaps, labels_all = amaps[:, 0], labels_all[:, 0] # both b,h,w
        pixel_auroc = roc_auc_score(labels_all.flatten(), amaps.flatten()) # pixel score
        image_auroc = roc_auc_score(labels_all.amax((-1,-2)), amaps.amax((-1,-2))) # image score
        pixel_pro = compute_pro_score_fast(amaps, labels_all) # pro score

        if pixel_auroc > np.max(pixel_auroc_lst):
            save_dict['state_dict_pixel'] = {k: v.cpu() for k, v in flow.state_dict().items()} # save ckpt
            save_dict['feat_mean'] = feat_mean_to_numpy_array(feat_mean)
            save_dict['pixel_threshold'] = pixel_threshold
            save_dict['image_threshold'] = image_threshold
            save_dict['pixel_threshold_percentile'] = pixel_threshold_percentile
            save_dict['image_threshold_percentile'] = image_threshold_percentile
            save_dict['threshold_source'] = 'valid_good_percentile'
        if pixel_pro > np.max(pixel_pro_lst):
            save_dict['state_dict_pro'] = {k: v.cpu() for k, v in flow.state_dict().items()} # save ckpt
            save_dict['pixel_threshold_pro'] = pixel_threshold
            save_dict['image_threshold_pro'] = image_threshold
        pixel_auroc_lst.append(pixel_auroc)
        pixel_pro_lst.append(pixel_pro)
        image_auroc_lst.append(image_auroc)

        # MOAI Logs
        twriter.add_scalar("train/loss", mean_loss, epoch)
        twriter.add_scalar("train/base_loss", mean_base_loss, epoch)
        twriter.add_scalar("train/continuity_loss", mean_continuity_loss, epoch)
        twriter.add_scalar("train/continuity_alpha", continuity_alpha, epoch)
        twriter.add_scalar("train/continuity_beta", continuity_beta, epoch)
        twriter.add_scalar("train/loss_mode", loss_mode, epoch)
        twriter.add_scalar("threshold/pixel", pixel_threshold, epoch)
        twriter.add_scalar("threshold/image", image_threshold, epoch)
        remaining = remaining_time(t0, epoch, epoches)
        result_csv.writerow([
            epoch,
            remaining,
            pixel_auroc,
            image_auroc,
            pixel_pro,
            mean_loss,
            mean_base_loss,
            mean_continuity_loss,
            continuity_alpha,
            continuity_beta,
            loss_mode,
            pixel_threshold,
            image_threshold,
        ])
        result_file.flush()
        
        # print
        print(f'[epoch: {epoch}/{epoches}] - remaining time: {remaining}s')
        print(f'   Loss: {mean_loss}')
        print(f'   Base-Loss: {mean_base_loss}')
        print(f'   Continuity-Loss: {mean_continuity_loss}')
        print(f'   Continuity-Alpha: {continuity_alpha}')
        print(f'   Continuity-Beta: {continuity_beta}')
        print(f'   Loss-Mode: {loss_mode}')
        print(f'   Pixel-AUROC: {pixel_auroc}')
        print(f'   Image-AUROC: {image_auroc}')
        print(f'   Pixel-PRO: {pixel_pro}')
        print(f'   Pixel-Threshold: {pixel_threshold}')
        print(f'   Image-Threshold: {image_threshold}')

        log_data = {
            "epoch": epoch,
            "time": remaining,
            "pixel_auroc": pixel_auroc,
            "image_auroc": image_auroc,
            "pixel_pro": pixel_pro,
            "loss": mean_loss,
            "base_loss": mean_base_loss,
            "continuity_loss": mean_continuity_loss,
            "continuity_alpha": continuity_alpha,
            "continuity_beta": continuity_beta,
            "loss_mode": loss_mode,
            "pixel_threshold": pixel_threshold,
            "image_threshold": image_threshold
        }
        message = {
            "status": "in_progress",
            "message": log_data
        }
        print(json.dumps(message, ensure_ascii=False), flush=True)

        del amaps, labels_all, diff_list, labels_list

    save_dict['pixel_auroc_lst'] = pixel_auroc_lst
    save_dict['image_auroc_lst'] = image_auroc_lst
    save_dict['pixel_pro_lst']   = pixel_pro_lst
    save_dict['losses_lst'] = losses_lst

    np.savez(f'{run_mpfm.weight_path}/best.npz', **save_dict) # save all 

    message = {
        "status": "complete",
        "message": "모델 학습이 성공적으로 완료되었습니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def pyramid_continuity_loss(pyramid, alpha=0.25, beta=0.5, mode="l1"):
    losses = [diff_loss_2d(feat, alpha=alpha, beta=beta, mode=mode) for feat in pyramid]
    return sum(losses) / len(losses)


def estimate_thresholds(flow, val_loader, feat_mean, device, pixel_percentile=99.5, image_percentile=95.0):
    normal_maps = []
    flow.eval()
    with torch.no_grad():
        for val_dict in val_loader:
            image = val_dict['images'].to(device)
            pyramid2 = flow(image)
            pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid2, feat_mean)]
            diff = flow.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)
            normal_maps.append(diff.cpu())

    normal_maps = torch.concat(normal_maps, dim=0)[:, 0]
    pixel_q = max(0.0, min(1.0, float(pixel_percentile) / 100.0))
    image_q = max(0.0, min(1.0, float(image_percentile) / 100.0))
    pixel_threshold = torch.quantile(normal_maps.flatten(), pixel_q).item()
    image_threshold = torch.quantile(normal_maps.amax((-1, -2)), image_q).item()
    return float(pixel_threshold), float(image_threshold)


def feat_mean_to_numpy_array(feat_mean):
    feat_mean_np = [f.cpu().numpy() for f in feat_mean]
    feat_mean_arr = np.empty(len(feat_mean_np), dtype=object)
    for i, arr in enumerate(feat_mean_np):
        feat_mean_arr[i] = arr
    return feat_mean_arr


def savefm(feat_mean, run_mpfm):
    feat_mean_arr = feat_mean_to_numpy_array(feat_mean)

    np.savez(f'{run_mpfm.weight_path}/feat_mean.npz', feat_mean=feat_mean_arr)


def get_vn_dims(volume_norm):
    if volume_norm == 'CVN':
        return (0, 1)
    if volume_norm == 'SVN':
        return (0, 2, 3)
    raise ValueError(f"Unknown volumeNorm: {volume_norm}")


if __name__ == '__main__':
    jobs = resolve_dataset_jobs(opt.dataset, opt.category, opt.dataset_root)

    for train_path, val_path, cls_name, version in jobs:
        run_mpfm, run_hyp, result_file, result_csv, twriter = prepare_run(version)
        try:
            vn_dims = get_vn_dims(run_hyp.volumeNorm)
            resnetX = 0 if run_hyp.encoder=='none' else int(run_hyp.encoder[6:])
            print(f"[category] {cls_name} -> version {version}")
            train(train_path, val_path,\
                  resnetX, run_hyp.numLayer, vn_dims, \
                  ksize=run_hyp.kernelSize, channel=run_hyp.numChannel, num_stack=run_hyp.numStack, \
                  device=f'cuda:{run_hyp.gpu}', batch_size=run_hyp.batchSize, epoches=run_hyp.epoches, hyp=run_hyp,
                  run_mpfm=run_mpfm, result_file=result_file, result_csv=result_csv, twriter=twriter, cls_name=cls_name)
        finally:
            twriter.close()
            if not result_file.closed:
                result_file.close()
