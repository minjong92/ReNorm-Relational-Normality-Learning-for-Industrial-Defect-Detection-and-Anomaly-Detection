#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os

import torch
from torchvision import transforms

import efficientad as ead
from common import ImageFolderWithoutTarget, ImageFolderWithPath


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help='Existing run directory to evaluate.')
    parser.add_argument('--dataset_path',
                        default='E:/0_dataset/mvtec_anomaly_detection',
                        help='MVTec AD dataset path.')
    parser.add_argument('-s', '--subdataset', default='all',
                        help='Sub-dataset name, or "all".')
    parser.add_argument('--target_fpr', type=float, default=0.001,
                        help='Target validation false positive rate. '
                             '0.001 means 0.1%%.')
    parser.add_argument('--cont_score_w', type=float, default=1.0,
                        help='Relative weight of the continuity anomaly map '
                             'when the run has a trained continuity head. '
                             '1.0 = equal thirds, 0.0 = original 0.5/0.5. '
                             'Ignored for runs without a continuity head.')
    return parser.parse_args()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_models(train_output_dir, device):
    teacher = torch_load(os.path.join(train_output_dir, 'teacher_final.pth'),
                         device)
    student = torch_load(os.path.join(train_output_dir, 'student_final.pth'),
                         device)
    autoencoder = torch_load(
        os.path.join(train_output_dir, 'autoencoder_final.pth'), device)
    normalization = torch_load(os.path.join(train_output_dir,
                                            'normalization.pt'), device)

    teacher.eval()
    student.eval()
    autoencoder.eval()
    if ead.on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    # continuity 점수 헤드는 continuity 학습 런에서만 저장된다. 없으면 None.
    head = None
    head_path = os.path.join(train_output_dir, 'continuity_head_final.pth')
    if os.path.exists(head_path):
        head = torch_load(head_path, device)
        head.eval()
        if ead.on_gpu:
            head.cuda()

    return teacher, student, autoencoder, normalization, head


def validation_split(dataset_path, subdataset):
    # MVTec LOCO는 전용 validation/ 폴더가 있으면 그것을 사용한다.
    val_dir = os.path.join(dataset_path, subdataset, 'validation')
    if os.path.isdir(val_dir):
        return ImageFolderWithoutTarget(
            val_dir, transform=transforms.Lambda(ead.train_transform))
    # MVTec AD: train의 10%를 검증셋으로 분할 (학습과 동일 규약)
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, subdataset, 'train'),
        transform=transforms.Lambda(ead.train_transform))
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(ead.seed)
    train_set, validation_set = torch.utils.data.random_split(
        full_train_set, [train_size, validation_size], rng)
    return validation_set


def continuity_run_present(config, subdatasets):
    """런에 continuity head가 저장돼 있으면 True."""
    for subdataset in subdatasets:
        head_path = os.path.join(config.run_dir, 'trainings', subdataset,
                                 'continuity_head_final.pth')
        if os.path.exists(head_path):
            return True
    return False

def eval_output_base(config, has_continuity):
    """continuity 런은 비중별로 결과를 분리 저장(eval_w<비중>/), 원본 런은 run_dir 그대로."""
    if has_continuity:
        return os.path.join(config.run_dir,
                            'eval_w{:g}'.format(config.cont_score_w))
    return config.run_dir

def evaluate_subdataset(config, subdataset, device, output_base):
    train_output_dir = os.path.join(config.run_dir, 'trainings', subdataset)
    metrics_output_dir = os.path.join(output_base, 'metrics', subdataset)
    os.makedirs(metrics_output_dir, exist_ok=True)

    teacher, student, autoencoder, normalization, head = load_models(
        train_output_dir, device)
    teacher_mean = normalization['teacher_mean'].to(device)
    teacher_std = normalization['teacher_std'].to(device)
    q_st_start = normalization['q_st_start'].to(device)
    q_st_end = normalization['q_st_end'].to(device)
    q_ae_start = normalization['q_ae_start'].to(device)
    q_ae_end = normalization['q_ae_end'].to(device)

    # continuity head와 그 정규화 분위가 둘 다 있어야 map_cont를 반영한다.
    q_cont_start = q_cont_end = None
    if head is not None and 'q_cont_start' in normalization:
        q_cont_start = normalization['q_cont_start'].to(device)
        q_cont_end = normalization['q_cont_end'].to(device)
    else:
        head = None  # 정규화 분위가 없으면 원본(st+ae)로 폴백
    print('{}: continuity {} (cont_score_w={})'.format(
        subdataset, 'ON' if head is not None else 'OFF', config.cont_score_w))

    validation_set = validation_split(config.dataset_path, subdataset)
    threshold, threshold_quantile, validation_scores = ead.validation_threshold(
        validation_set=validation_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        target_fpr=config.target_fpr, head=head, q_cont_start=q_cont_start,
        q_cont_end=q_cont_end, cont_score_w=config.cont_score_w)
    validation_metrics = ead.image_classification_metrics(
        [0] * len(validation_scores), validation_scores, threshold)

    test_set = ImageFolderWithPath(
        os.path.join(config.dataset_path, subdataset, 'test'))
    auc, rows, test_metrics = ead.test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=None, threshold=threshold,
        scores_output_path=os.path.join(metrics_output_dir,
                                        'image_scores.csv'),
        desc='{} inference'.format(subdataset),
        head=head, q_cont_start=q_cont_start, q_cont_end=q_cont_end,
        cont_score_w=config.cont_score_w)

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
    ead.write_metrics_csv(os.path.join(metrics_output_dir,
                                       'image_metrics.csv'), [metrics_row])
    print('{}: AUROC={:.4f}, threshold={:.8f}, TP={} TN={} FP={} FN={}, '
          'mean={:.4f} ms, fps={:.2f}'.format(
              subdataset, auc, threshold, test_metrics['tp'],
              test_metrics['tn'], test_metrics['fp'], test_metrics['fn'],
              test_metrics['inference_mean_ms'],
              test_metrics['inference_fps']))
    return metrics_row


def run_subdatasets(config):
    if config.subdataset == 'all':
        subdatasets = ead.list_subdatasets(config.dataset_path)
    else:
        subdatasets = [config.subdataset]

    has_continuity = continuity_run_present(config, subdatasets)
    output_base = eval_output_base(config, has_continuity)
    if has_continuity:
        print('Continuity run detected. cont_score_w={} -> saving to {}'.format(
            config.cont_score_w, output_base))
    else:
        print('No continuity head found -> original st+ae eval, saving to {}'.format(
            output_base))

    device = 'cuda' if ead.on_gpu else 'cpu'
    rows = []
    for subdataset in subdatasets:
        rows.append(evaluate_subdataset(config, subdataset, device, output_base))
    ead.write_metrics_csv(os.path.join(output_base, 'metrics_summary.csv'),
                          rows)


if __name__ == '__main__':
    config = get_argparse()
    ead.validate_config(config)
    run_subdatasets(config)
