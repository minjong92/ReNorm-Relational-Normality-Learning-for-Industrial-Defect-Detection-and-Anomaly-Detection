#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import csv
import itertools
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from continuity import ContinuityHead, npp_loss

def get_argparse():
    parser = argparse.ArgumentParser()
    ###########################################################################
    # main arguments
    parser.add_argument('--continuity_test', type=int, default=1,
                        choices=[0, 1],
                        help='0: original EfficientAD. '
                             '1: add continuity (next-patch-prediction) loss; '
                             'map_cont is mixed into the score per --cont_score_w '
                             '(0 = train only, ignore at inference)')
    parser.add_argument('--cont_train_w', type=float, default=1.0,
                        help='Weight of the continuity loss added to the '
                             'training total (only used when --continuity_test 1)')
    parser.add_argument('--cont_score_w', type=float, default=1.0,
                        help='Relative weight of the continuity anomaly map when '
                             'combined with map_st/map_ae. 1.0 = equal thirds, '
                             '0.0 = original 0.5/0.5 (train head but ignore at '
                             'inference)')
    parser.add_argument('--cont_loss_type', default='next',
                        choices=['next', 'diff', 'next+diff'],
                        help='Continuity loss component to use.')
    parser.add_argument('--cont_dropout', type=float, default=0.0,
                        help='Feature-dropout rate in [0,1) applied to the '
                             'predictor INPUT only (target stays clean). '
                             '0 disables. Active during training only.')
    parser.add_argument('--cont_alpha', type=float, default=0.0,
                        help='First-difference loss weight for continuity. '
                             '0 keeps the previous next-prediction-only loss.')
    parser.add_argument('--cont_beta', type=float, default=0.0,
                        help='Second-difference loss weight for continuity. '
                             '0 keeps the previous next-prediction-only loss.')
    parser.add_argument('--npp_mode', default='l1',
                        choices=['l1', 'l2'],
                        help='Distance mode for the continuity prediction '
                             'residual (l1 recommended)')
    ##########################################################################
    # dataset arguments
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', nargs='+', default=['all'],
                        help="Sub-dataset name(s), or 'all' for every "
                             "sub-dataset. Multiple allowed: -s capsule pill")
    ##########################################################################
    # train arguments
    parser.add_argument('-o', '--output_dir', default='results')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-t', '--train_steps', type=int, default=10000)
    parser.add_argument('--target_fpr', type=float, default=0.001,
                        help='Target validation false positive rate for the '
                             'image-level threshold. 0.001 means 0.1%%.')
    ##########################################################################
    # etc. arguments
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='E:/0_dataset/mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='E:/0_dataset/mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    ##########################################################################
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

METRIC_FIELDS = [
    'subdataset', 'image_auroc', 'threshold', 'threshold_quantile',
    'target_fpr', 'validation_images', 'validation_fp', 'validation_fpr',
    'test_images', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall',
    'specificity', 'fpr', 'fnr', 'f1', 'balanced_accuracy',
    'inference_total_sec', 'inference_mean_ms', 'inference_median_ms',
    'inference_p95_ms', 'inference_fps'
]

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))


def dataset_output_name(dataset_path):
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    if not dataset_name:
        raise ValueError('Invalid dataset path: {}'.format(dataset_path))
    return dataset_name

def create_run_output_dir(output_dir, dataset_path):
    dataset_output_dir = os.path.join(output_dir,
                                      dataset_output_name(dataset_path))
    timestamp = datetime.now().strftime('%y%m%d%H%M')
    run_output_dir = os.path.join(dataset_output_dir, timestamp)
    suffix = 1
    while os.path.exists(run_output_dir):
        run_output_dir = os.path.join(dataset_output_dir, '{}_{:02d}'.format(
            timestamp, suffix))
        suffix += 1
    return run_output_dir

def hard_mse_loss(teacher_output, student_output):
    distance = (teacher_output - student_output) ** 2
    threshold = torch.quantile(distance, q=0.999)
    return torch.mean(distance[distance >= threshold])

def loss_status_text(continuity_test):
    statuses = {
        0: '원본 EfficientAD(continuity 미사용)',
        1: 'continuity 다음-패치 예측 점수 분기 추가',
    }
    return statuses[continuity_test]

def write_loss_status(run_output_dir, config):
    os.makedirs(run_output_dir, exist_ok=True)
    path = os.path.join(run_output_dir, 'loss_status.txt')
    with open(path, 'w', encoding='utf-8') as file:
        file.write(loss_status_text(config.continuity_test) + '\n')

def write_run_arguments(run_output_dir, config):
    """실행에 사용된 명령어와 모든 인자를 arguments.txt로 기록."""
    os.makedirs(run_output_dir, exist_ok=True)
    path = os.path.join(run_output_dir, 'arguments.txt')
    with open(path, 'w', encoding='utf-8') as file:
        file.write('timestamp: {}\n'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        file.write('command: {}\n'.format(' '.join(sys.argv)))
        file.write('loss_status: {}\n'.format(
            loss_status_text(config.continuity_test)))
        file.write('\n[arguments]\n')
        for key, value in sorted(vars(config).items()):
            file.write('{}: {}\n'.format(key, value))

def save_normalization(train_output_dir, teacher_mean, teacher_std,
                       q_st_start, q_st_end, q_ae_start, q_ae_end,
                       q_cont_start=None, q_cont_end=None):
    payload = {
        'teacher_mean': teacher_mean.detach().cpu(),
        'teacher_std': teacher_std.detach().cpu(),
        'q_st_start': q_st_start.detach().cpu(),
        'q_st_end': q_st_end.detach().cpu(),
        'q_ae_start': q_ae_start.detach().cpu(),
        'q_ae_end': q_ae_end.detach().cpu(),
        'image_size': image_size,
        'out_channels': out_channels,
    }
    if q_cont_start is not None:
        payload['q_cont_start'] = q_cont_start.detach().cpu()
        payload['q_cont_end'] = q_cont_end.detach().cpu()
    torch.save(payload, os.path.join(train_output_dir, 'normalization.pt'))

def set_seed():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_dataset_path(config):
    if config.dataset == 'mvtec_ad':
        return config.mvtec_ad_path
    if config.dataset == 'mvtec_loco':
        return config.mvtec_loco_path
    raise ValueError('Unknown dataset: {}'.format(config.dataset))

def list_subdatasets(dataset_path):
    subdatasets = []
    for name in sorted(os.listdir(dataset_path)):
        subdataset_path = os.path.join(dataset_path, name)
        if not os.path.isdir(subdataset_path):
            continue
        if (os.path.isdir(os.path.join(subdataset_path, 'train')) and
                os.path.isdir(os.path.join(subdataset_path, 'test'))):
            subdatasets.append(name)
    if not subdatasets:
        raise ValueError('No sub-datasets found in {}'.format(dataset_path))
    return subdatasets

def resolve_subdatasets(requested, dataset_path):
    """요청 이름(하나/여러개/'all')을 실제 서브데이터셋 폴더 목록으로 해석.

    'all'이 포함되면 경로의 모든 서브데이터셋을, 아니면 요청한 이름들을
    그대로 쓰되 폴더가 존재하는지 검증한다.
    """
    available = list_subdatasets(dataset_path)
    if 'all' in requested:
        return available
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError('Unknown sub-dataset(s) {} under {}. Available: {}'
                         .format(unknown, dataset_path, available))
    return list(requested)

def validate_config(config):
    # evaluate_existing_run 등 cont_* 인자가 없는 호출자도 안전하도록 getattr 사용.
    if config.target_fpr < 0 or config.target_fpr >= 1:
        raise ValueError('--target_fpr must be in the range [0, 1).')
    if getattr(config, 'cont_dropout', 0.0) < 0 or getattr(config, 'cont_dropout', 0.0) > 1:
        raise ValueError('--cont_dropout must be in the range [0, 1].')
    if getattr(config, 'cont_alpha', 0.0) < 0:
        raise ValueError('--cont_alpha must be >= 0.')
    if getattr(config, 'cont_beta', 0.0) < 0:
        raise ValueError('--cont_beta must be >= 0.')

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0

def format_csv_value(value):
    if isinstance(value, float):
        return '{:.8f}'.format(value)
    return value

def write_metrics_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: format_csv_value(row.get(field, ''))
                for field in METRIC_FIELDS
            })

def write_image_scores(path, rows):
    fieldnames = [
        'path', 'defect_class', 'image', 'label', 'score', 'prediction',
        'inference_ms'
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'path': row['path'],
                'defect_class': row['defect_class'],
                'image': row['image'],
                'label': row['label'],
                'score': format_csv_value(row['score']),
                'prediction': row.get('prediction', ''),
                'inference_ms': format_csv_value(row.get('inference_ms', '')),
            })

def image_classification_metrics(y_true, y_score, threshold):
    # Scores equal to the validation threshold stay normal. This keeps the
    # empirical validation FPR conservative for small validation sets.
    y_pred = [1 if score > threshold else 0 for score in y_score]
    tp = sum(1 for true, pred in zip(y_true, y_pred)
             if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred)
             if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred)
             if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred)
             if true == 1 and pred == 0)

    accuracy = safe_divide(tp + tn, len(y_true))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, fp + tn)
    fnr = safe_divide(fn, fn + tp)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        'test_images': len(y_true),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr,
        'f1': f1,
        'balanced_accuracy': 0.5 * (recall + specificity),
    }

def inference_time_metrics(inference_times):
    if not inference_times:
        return {
            'inference_total_sec': 0.0,
            'inference_mean_ms': 0.0,
            'inference_median_ms': 0.0,
            'inference_p95_ms': 0.0,
            'inference_fps': 0.0,
        }
    times = np.asarray(inference_times, dtype=np.float64)
    total_sec = float(np.sum(times))
    return {
        'inference_total_sec': total_sec,
        'inference_mean_ms': float(np.mean(times) * 1000.0),
        'inference_median_ms': float(np.median(times) * 1000.0),
        'inference_p95_ms': float(np.percentile(times, 95) * 1000.0),
        'inference_fps': safe_divide(len(times), total_sec),
    }

def threshold_from_normal_scores(normal_scores, target_fpr):
    if not normal_scores:
        raise ValueError('Cannot compute threshold without validation scores.')
    quantile = 1.0 - target_fpr
    scores = np.asarray(normal_scores, dtype=np.float64)
    try:
        threshold = np.quantile(scores, quantile, method='higher')
    except TypeError:
        threshold = np.quantile(scores, quantile, interpolation='higher')
    return float(threshold), quantile

def iter_original_images(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = range(len(base_dataset.samples))

    for index in indices:
        path, _ = base_dataset.samples[index]
        yield base_dataset.loader(path), path

def main():
    config = get_argparse()
    validate_config(config)
    dataset_path = get_dataset_path(config)

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    run_output_dir = create_run_output_dir(config.output_dir, dataset_path)
    write_loss_status(run_output_dir, config)
    write_run_arguments(run_output_dir, config)
    subdatasets = resolve_subdatasets(config.subdataset, dataset_path)
    print('Saving results to {}'.format(run_output_dir))
    print('Loss status: {}'.format(loss_status_text(config.continuity_test)))
    print('Running sub-datasets: {}'.format(', '.join(subdatasets)))

    summary_rows = []
    for index, subdataset in enumerate(subdatasets, start=1):
        print('[{}/{}] Running sub-dataset: {}'.format(
            index, len(subdatasets), subdataset))
        metrics_row = run_subdataset(
            config=config, dataset_path=dataset_path,
            run_output_dir=run_output_dir, subdataset=subdataset,
            pretrain_penalty=pretrain_penalty)
        summary_rows.append(metrics_row)
    write_metrics_csv(os.path.join(run_output_dir, 'metrics_summary.csv'),
                      summary_rows)

def run_subdataset(config, dataset_path, run_output_dir, subdataset,
                   pretrain_penalty):
    set_seed()

    train_output_dir = os.path.join(run_output_dir, 'trainings', subdataset)
    test_output_dir = os.path.join(run_output_dir, 'anomaly_maps', subdataset,
                                   'test')
    metrics_output_dir = os.path.join(run_output_dir, 'metrics', subdataset)
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)
    os.makedirs(metrics_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, subdataset, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise ValueError('Unknown dataset: {}'.format(config.dataset))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise ValueError('Unknown model_size: {}'.format(config.model_size))
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # continuity (다음-패치 예측) 점수 분기. student feature에 regularization을
    # 추가해 teacher를 모방하면서도 정상 feature가 공간적으로 안정되도록 한다.
    continuity_on = config.continuity_test >= 1
    if continuity_on:
        # 헤드의 무작위 초기화가 전역 RNG를 소비해 학습 데이터 셔플 순서를
        # 바꾸지 않도록 RNG 상태를 보존/복원한다.
        rng_state = torch.get_rng_state()
        continuity_head = ContinuityHead(
            in_ch=out_channels, mode=config.npp_mode,
            alpha=config.cont_alpha, beta=config.cont_beta,
            input_dropout=config.cont_dropout)
        # 사영층 고정(무작위 사영) → 상수 임베딩 붕괴 자명해 제거. 예측기만 학습.
        continuity_head.proj.requires_grad_(False)
        torch.set_rng_state(rng_state)
    else:
        continuity_head = None

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()
    if continuity_on:
        continuity_head.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        if continuity_on:
            continuity_head.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    trainable_params = itertools.chain(student.parameters(),
                                       autoencoder.parameters())
    if continuity_on:
        trainable_params = itertools.chain(
            trainable_params,
            (p for p in continuity_head.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        #########################################################################
        # Student-Teacher Loss(loss_st)
        with torch.no_grad():   # teacher는 고정
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        loss_hard = hard_mse_loss(teacher_output_st, student_output_st)
        #########################################################################
        # 패널티 loss(loss_penalty). student가 이상 이미지에 대해서는 teacher을 따라하지 못하게 하기 위해
        if image_penalty is not None:   # image_penalty는 외부 일반 이미지(ImageNet 같은)
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard
        #########################################################################
        # Autoencoder Loss(loss_ae). ae가 teacher feature를 복원하도록 학습됨
        # image_ae는 color jitter가 들어간 augmentation 이미지
        # 목표는 autoencoder(image_ae)와 teacher(image_ae) feature와 같아지는 것
        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

        distance_ae = (teacher_output_ae - ae_output)**2
        loss_ae = torch.mean(distance_ae)
        #######################################################################
        # student-autoencoder loss(loss_stae). student 출력의 뒤 절반은 autoencoder output을 따라하도록 학습.
        # 즉 앞 절반은 teacher imitation용, 뒤 절반은 autoencoder imitaion용
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_stae = (ae_output - student_output_ae)**2
        loss_stae = torch.mean(distance_stae)
        #######################################################################
        # continuity loss(loss_cont). student feature의 다음 패치를 예측하도록
        # 점수 헤드를 학습하고, student feature에도 연속성 regularization을 건다.
        loss_total = loss_st + loss_ae + loss_stae
        if continuity_on:
            # cont_dropout은 헤드 입력 feature dropout으로 적용됨(타깃은 clean).
            # 여기선 손실 위치 마스크 없이 전체 패치에서 채점.
            loss_cont = npp_loss(
                continuity_head, student_output_st, norm_mask=None,
                mode=config.npp_mode, alpha=config.cont_alpha,
                beta=config.cont_beta, loss_type=config.cont_loss_type)
            loss_total = loss_total + config.cont_train_w * loss_cont

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            description = "Current loss: {:.4f}  ".format(loss_total.item())
            tqdm_obj.set_description(description)

        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                             'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                             'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                 'autoencoder_tmp.pth'))
            if continuity_on:
                torch.save(continuity_head, os.path.join(
                    train_output_dir, 'continuity_head_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            teacher.eval()
            student.eval()
            autoencoder.eval()
            if continuity_on:
                continuity_head.eval()

            (q_st_start, q_st_end, q_ae_start, q_ae_end,
             q_cont_start, q_cont_end) = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                head=continuity_head, desc='Intermediate map normalization')
            auc, _, _ = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                head=continuity_head, q_cont_start=q_cont_start,
                q_cont_end=q_cont_end, cont_score_w=config.cont_score_w,
                test_output_dir=None, desc='Intermediate inference')
            print('{} intermediate image auc: {:.4f}'.format(subdataset, auc))

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()
            if continuity_on:
                continuity_head.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()
    if continuity_on:
        continuity_head.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))
    if continuity_on:
        torch.save(continuity_head, os.path.join(
            train_output_dir, 'continuity_head_final.pth'))

    (q_st_start, q_st_end, q_ae_start, q_ae_end,
     q_cont_start, q_cont_end) = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, head=continuity_head,
        desc='Final map normalization')
    save_normalization(train_output_dir, teacher_mean, teacher_std,
                       q_st_start, q_st_end, q_ae_start, q_ae_end,
                       q_cont_start, q_cont_end)
    threshold, threshold_quantile, validation_scores = validation_threshold(
        validation_set=validation_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        target_fpr=config.target_fpr, head=continuity_head,
        q_cont_start=q_cont_start, q_cont_end=q_cont_end,
        cont_score_w=config.cont_score_w)
    validation_metrics = image_classification_metrics(
        [0] * len(validation_scores), validation_scores, threshold)
    auc, test_rows, test_metrics = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        head=continuity_head, q_cont_start=q_cont_start,
        q_cont_end=q_cont_end, cont_score_w=config.cont_score_w,
        test_output_dir=test_output_dir, threshold=threshold,
        scores_output_path=os.path.join(metrics_output_dir,
                                        'image_scores.csv'),
        desc='Final inference')
    metrics_row = {
        'subdataset': subdataset,
        'image_auroc': auc,
        'threshold': threshold,
        'threshold_quantile': threshold_quantile,
        'target_fpr': config.target_fpr,
        'validation_images': len(validation_scores),
        'validation_fp': validation_metrics['fp'],
        'validation_fpr': validation_metrics['fpr'],
    }
    metrics_row.update(test_metrics)
    write_metrics_csv(os.path.join(metrics_output_dir, 'image_metrics.csv'),
                      [metrics_row])
    print('{} final image auc: {:.4f}'.format(subdataset, auc))
    print('{} threshold @ validation FPR {:.4%}: {:.8f}'.format(
        subdataset, config.target_fpr, threshold))
    print('{} test metrics: TP={} TN={} FP={} FN={} FPR={:.4%} FNR={:.4%}'.format(
        subdataset, test_metrics['tp'], test_metrics['tn'],
        test_metrics['fp'], test_metrics['fn'], test_metrics['fpr'],
        test_metrics['fnr']))
    print('{} inference: mean={:.4f} ms, p95={:.4f} ms, fps={:.2f}'.format(
        subdataset, test_metrics['inference_mean_ms'],
        test_metrics['inference_p95_ms'], test_metrics['inference_fps']))
    return metrics_row

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         threshold=None, scores_output_path=None, desc='Running inference',
         head=None, q_cont_start=None, q_cont_end=None, cont_score_w=1.0):
    y_true = []
    y_score = []
    rows = []
    inference_times = []
    for image, target, path in tqdm(test_set, desc=desc):
        if on_gpu:
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        map_combined = image_anomaly_map(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end, head=head,
            q_cont_start=q_cont_start, q_cont_end=q_cont_end,
            cont_score_w=cont_score_w)
        if on_gpu:
            torch.cuda.synchronize()
        inference_sec = time.perf_counter() - inference_start
        inference_times.append(inference_sec)

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
        row = {
            'path': path,
            'defect_class': defect_class,
            'image': os.path.basename(path),
            'label': y_true_image,
            'score': float(y_score_image),
            'inference_ms': inference_sec * 1000.0,
        }
        if threshold is not None:
            row['prediction'] = 1 if y_score_image > threshold else 0
        rows.append(row)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    metrics = (image_classification_metrics(y_true, y_score, threshold)
               if threshold is not None else None)
    if metrics is not None:
        metrics.update(inference_time_metrics(inference_times))
    if scores_output_path is not None:
        write_image_scores(scores_output_path, rows)
    return auc * 100, rows, metrics

@torch.no_grad()
def image_anomaly_map(image, teacher, student, autoencoder, teacher_mean,
                      teacher_std, q_st_start, q_st_end, q_ae_start,
                      q_ae_end, head=None, q_cont_start=None, q_cont_end=None,
                      cont_score_w=1.0):
    orig_width = image.width
    orig_height = image.height
    image = default_transform(image)
    image = image[None]
    if on_gpu:
        image = image.cuda()
    map_combined, map_st, map_ae, map_cont = predict(
        image=image, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end, head=head,
        q_cont_start=q_cont_start, q_cont_end=q_cont_end,
        cont_score_w=cont_score_w)
    map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
    map_combined = torch.nn.functional.interpolate(
        map_combined, (orig_height, orig_width), mode='bilinear')
    return map_combined[0, 0].cpu().numpy()

@torch.no_grad()
def validation_threshold(validation_set, teacher, student, autoencoder,
                         teacher_mean, teacher_std, q_st_start, q_st_end,
                         q_ae_start, q_ae_end, target_fpr, head=None,
                         q_cont_start=None, q_cont_end=None, cont_score_w=1.0):
    scores = []
    for image, path in tqdm(
            iter_original_images(validation_set),
            total=len(validation_set),
            desc='Computing validation threshold'):
        map_combined = image_anomaly_map(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start,
            q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            head=head, q_cont_start=q_cont_start, q_cont_end=q_cont_end,
            cont_score_w=cont_score_w)
        scores.append(float(np.max(map_combined)))
    threshold, quantile = threshold_from_normal_scores(scores, target_fpr)
    return threshold, quantile, scores

@torch.no_grad()
def continuity_anomaly_map(head, feature_output):
    """다음-패치 예측 잔차를 [B,1,H,W] 이상맵으로 복원.

    head는 z1=z[:,1:], z1_hat=z_hat[:,:-1]을 돌려주므로 잔차는 위치 1..T-1에 대응.
    위치 0은 문맥이 없어 0, 행 경계(t%W==0, 좌측 열)는 래스터 래핑이라 제거.
    """
    height, width = feature_output.shape[-2], feature_output.shape[-1]
    z1, z1_hat, _ = head(feature_output)
    residual = ((z1_hat - z1) ** 2).mean(-1)              # [B, T-1]
    amap = residual.new_zeros(residual.size(0), height * width)
    amap[:, 1:] = residual
    amap = amap.view(residual.size(0), 1, height, width)
    amap[:, :, :, 0] = 0                                  # 행 경계 + 위치 0 제거
    return amap

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None,
            head=None, q_cont_start=None, q_cont_end=None, cont_score_w=1.0):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    map_cont = None
    if head is not None:
        map_cont = continuity_anomaly_map(head, student_output[:, :out_channels])
        if q_cont_start is not None:
            map_cont = 0.1 * (map_cont - q_cont_start) / (q_cont_end - q_cont_start)

    if map_cont is not None:
        total_w = 2.0 + cont_score_w
        map_combined = (map_st + map_ae + cont_score_w * map_cont) / total_w
    else:
        map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae, map_cont

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, head=None,
                      desc='Map normalization'):
    maps_st = []
    maps_ae = []
    maps_cont = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae, map_cont = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, head=head)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
        if map_cont is not None:
            maps_cont.append(map_cont)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    q_cont_start = q_cont_end = None
    if maps_cont:
        maps_cont = torch.cat(maps_cont)
        q_cont_start = torch.quantile(maps_cont, q=0.9)
        q_cont_end = torch.quantile(maps_cont, q=0.995)
    return (q_st_start, q_st_end, q_ae_start, q_ae_end,
            q_cont_start, q_cont_end)

@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
