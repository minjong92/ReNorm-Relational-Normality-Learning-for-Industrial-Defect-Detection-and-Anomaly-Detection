import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse
import csv
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from model import PyramidFlow
from util import MOAIDataloader
from util import fix_randseed, compute_pro_score_fast

from pathfilemgr import MPathFileManager
from hyp_data import MHyp

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')
parser.add_argument('--dataset', help='dataset folder name under ./dataset, e.g. mvtec_anomaly_detection')
parser.add_argument('--category', help='category folder name inside the dataset, e.g. bottle')
parser.add_argument('--dataset-root', help='local dataset root directory', default=r'E:\0_dataset')
parser.add_argument('--threshold', type=float, default=None, help='manual raw pixel threshold; default loads threshold from best.npz')
parser.add_argument('--image-threshold', type=float, default=None, help='manual raw image threshold; default loads threshold from best.npz')

opt = parser.parse_args()

def resolve_test_datapath(dataset_name, category_name, dataset_root, mpfm, cls_name=None):
    root = dataset_root if os.path.isabs(dataset_root) else os.path.join(os.getcwd(), dataset_root)

    if not dataset_name and cls_name and '/' in str(cls_name).replace('\\', '/'):
        parts = str(cls_name).replace('\\', '/').split('/')
        dataset_name = parts[0]
        category_name = parts[1] if len(parts) > 1 and not category_name else category_name

    if not dataset_name:
        return mpfm.test_dataset

    dataset_path = os.path.join(root, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    if os.path.isdir(os.path.join(dataset_path, 'test')):
        return dataset_path

    if not category_name:
        raise ValueError(f"Pass --category for dataset folder: {dataset_path}")

    category_path = os.path.join(dataset_path, category_name)
    if not os.path.isdir(category_path):
        raise FileNotFoundError(f"Category folder not found: {category_path}")
    return category_path


def is_category_root(path):
    return os.path.isdir(os.path.join(path, 'test')) or os.path.isdir(os.path.join(path, 'train', 'good'))


def list_dataset_categories(dataset_name, dataset_root):
    root = dataset_root if os.path.isabs(dataset_root) else os.path.join(os.getcwd(), dataset_root)
    dataset_path = os.path.join(root, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    if is_category_root(dataset_path):
        return []
    return [
        name for name in sorted(os.listdir(dataset_path))
        if os.path.isdir(os.path.join(dataset_path, name)) and is_category_root(os.path.join(dataset_path, name))
    ]


def list_trained_categories(base_mpfm):
    categories = []
    if not os.path.isdir(base_mpfm.version_path):
        return categories
    for name in sorted(os.listdir(base_mpfm.version_path)):
        weight_path = os.path.join(base_mpfm.version_path, name, 'weights', 'best.npz')
        if os.path.exists(weight_path):
            categories.append(name)
    return categories


def resolve_test_jobs():
    base_mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)

    if opt.category is not None and opt.category.lower() == 'all':
        if opt.dataset:
            categories = list_dataset_categories(opt.dataset, opt.dataset_root)
        else:
            categories = list_trained_categories(base_mpfm)

        if not categories:
            raise FileNotFoundError(
                "No categories found for --category all. "
                "Pass --dataset or check trained category folders under the requested version."
            )

        jobs = []
        for category in categories:
            jobs.append({
                'version': f'{opt.version}/{category}',
                'dataset': opt.dataset,
                'category': category,
            })
        return base_mpfm, jobs

    return base_mpfm, [{
        'version': opt.version,
        'dataset': opt.dataset,
        'category': opt.category,
    }]


def load_hyp_for_mpfm(run_mpfm):
    hyp = MHyp()
    if os.path.exists(run_mpfm.test_hyp_yaml):
        run_mpfm.load_test_hyp(hyp)
    elif os.path.exists(run_mpfm.hyp_yaml):
        run_mpfm.load_train_hyp(hyp)
    return hyp


def get_vn_dims(volume_norm):
    if volume_norm == 'CVN':
        return (0, 1)
    if volume_norm == 'SVN':
        return (0, 2, 3)
    raise ValueError(f"Unknown volumeNorm: {volume_norm}")


def safe_roc_auc(y_true, y_score):
    y_true_np = y_true.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    y_score_np = y_score.detach().cpu().numpy().reshape(-1)
    if np.unique(y_true_np).size < 2:
        return float('nan')
    return float(roc_auc_score(y_true_np, y_score_np))


def confusion_counts(y_true, y_pred):
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    tp = torch.logical_and(y_true, y_pred).sum().item()
    tn = torch.logical_and(~y_true, ~y_pred).sum().item()
    fp = torch.logical_and(~y_true, y_pred).sum().item()
    fn = torch.logical_and(y_true, ~y_pred).sum().item()
    return int(tn), int(fp), int(fn), int(tp)


def normalize_tensor(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def optional_np_scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    if np.isscalar(value):
        return value.item() if hasattr(value, 'item') else value
    return value.item()


def write_test_csvs(mpfm, metrics, per_image_rows):
    metrics_path = f'{mpfm.test_result}/metrics.csv'
    with open(metrics_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    per_image_path = f'{mpfm.test_result}/per_image_results.csv'
    with open(per_image_path, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = ['fname', 'image_label', 'image_score', 'image_score_norm', 'inference_time_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    confusion_path = f'{mpfm.test_result}/confusion_matrix.csv'
    with open(confusion_path, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = ['level', 'tn', 'fp', 'fn', 'tp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'level': 'pixel',
            'tn': metrics['pixel_tn'],
            'fp': metrics['pixel_fp'],
            'fn': metrics['pixel_fn'],
            'tp': metrics['pixel_tp'],
        })
        writer.writerow({
            'level': 'image',
            'tn': metrics['image_tn'],
            'fp': metrics['image_fp'],
            'fn': metrics['image_fn'],
            'tp': metrics['image_tp'],
        })

    return metrics_path, per_image_path, confusion_path


def write_summary_csv(base_mpfm, rows):
    summary_path = os.path.join(base_mpfm.test_result, 'test_summary.csv')
    if not rows:
        return summary_path

    preferred = [
        'version',
        'category',
        'cls_name',
        'num_images',
        'pixel_auroc',
        'image_auroc',
        'pixel_pro',
        'pixel_threshold',
        'image_threshold',
        'pixel_threshold_mode',
        'image_threshold_mode',
        'pixel_tn',
        'pixel_fp',
        'pixel_fn',
        'pixel_tp',
        'image_tn',
        'image_fp',
        'image_fn',
        'image_tp',
        'total_inference_time_ms',
        'avg_inference_time_ms',
        'fps',
        'test_datapath',
    ]
    extra = [key for key in rows[0].keys() if key not in preferred]
    fieldnames = preferred + extra

    with open(summary_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def test(
    resnetX,
    num_layer,
    vn_dims,
    ksize,
    channel,
    num_stack,
    device,
    mpfm,
    dataset_name=None,
    category_name=None,
    dataset_root='dataset',
    threshold=0.5,
    image_threshold=None,
):
    message = {
        "status": "prepare",
        "message": "추론 환경을 준비중입니다.",
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

    loader_dict = fix_randseed(seed=0)

    model_data = np.load(f'{mpfm.weight_path}/best.npz', allow_pickle=True)

    print(f'{mpfm.weight_path}/best.npz')

    cls_name = model_data['cls_name'].item() if 'cls_name' in model_data else 'moai'
    trained_pixel_threshold = optional_np_scalar(model_data, 'pixel_threshold')
    trained_image_threshold = optional_np_scalar(model_data, 'image_threshold')
    threshold_source = optional_np_scalar(model_data, 'threshold_source', 'manual_or_fallback')
    test_datapath = resolve_test_datapath(dataset_name, category_name, dataset_root, mpfm, cls_name=cls_name)
    resnetX = model_data['resnetX'] if np.isscalar(model_data['resnetX']) else model_data['resnetX'].item()
    num_layer = model_data['num_layer'] if np.isscalar(model_data['num_layer']) else model_data['num_layer'].item()
    vn_dims = (0, 1)# model_data['vn_dims'] # 튜플/리스트이므로 item()을 호출하지 않음
    ksize = model_data['ksize'] if np.isscalar(model_data['ksize']) else model_data['ksize'].item()
    channel = model_data['channel'] if np.isscalar(model_data['channel']) else model_data['channel'].item()
    num_stack = model_data['num_stack'] if np.isscalar(model_data['num_stack']) else model_data['num_stack'].item()
    #x_size = 256 if resnetX==0 else 1024
    #resnetX = 0 if resnetX=='none' else int(resnetX[6:])

    x_size = 256 if resnetX==0 else 1024

    print(f"클래스: {cls_name}")
    print(f"ResNet: {resnetX}")
    print(f"층 수: {num_layer}")
    print(f"정규화 차원: {vn_dims}")
    print(f"커널 크기: {ksize}")
    print(f"채널 수: {channel}")
    print(f"스택 수: {num_stack}")
    print(f"테스트 데이터셋: {test_datapath}")

    print(resnetX, channel, num_layer, num_stack, ksize, vn_dims, False)

    # model
    flow = PyramidFlow(resnetX, channel, num_layer, num_stack, ksize, vn_dims, False).to(device)

    # run dymmy on the model
    with torch.no_grad():
        dummy = torch.zeros((1, 3, x_size, x_size)).to(device)
        _ = flow(dummy)

    try:
        # 키가 있는지 확인
        if 'state_dict_pixel' in model_data:
            print(f"[I]Pixel AUROC 기준 최적 모델 사용 중")
            state_dict_key = 'state_dict_pixel'
        elif 'state_dict_pro' in model_data:
            print(f"[I]PRO Score 기준 최적 모델 사용 중")
            state_dict_key = 'state_dict_pro'
        else:
            raise ValueError("모델 상태를 찾을 수 없습니다")
        
        # state_dict 가져오기
        state_dict_data = model_data[state_dict_key].item()
        print(f"[I]상태 딕셔너리 키 개수: {len(state_dict_data)}")
            
        # 일부 키만 출력
        some_keys = list(state_dict_data.keys())[:5]
        print(f"[I]상태 딕셔너리 일부 키: {some_keys}")
            
        # 상태 딕셔너리 변환
        state_dict = {}
        for k, v in state_dict_data.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v
            else:
                state_dict[k] = torch.from_numpy(v)


        # 모델 가중치 로딩
        print("[I]모델 가중치 로딩 중...")
        missing_keys, unexpected_keys = flow.load_state_dict(state_dict, strict=False)
            
        if len(missing_keys) > 0:
            print(f"[I]누락된 키: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[I]예상치 못한 키: {unexpected_keys}")
                
    except Exception as e:
        #print(f"[E]모델 로딩 오류: {str(e)}")//server에서 Error 메세지를 받고 컨테이너를 종료시키므로...
        print("[E]가중치 로딩을 건너뛰고 계속 진행합니다.")



    # dataset
    #val_dataset = MOAIDataloader(mode='val', x_size=x_size, y_size=256, datapath=mpfm.train_path)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, drop_last=False, **loader_dict)
    test_dataset = MOAIDataloader(mode='test', x_size=x_size, y_size=256, datapath=test_datapath)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True, **loader_dict)

    message = {
        "status": "start",
        "message": "모델 추론을 시작합니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

    feat_mean = loadfm(device, mpfm, model_data)

    # test
    flow.eval()
    diff_list, labels_list = [], []
    inference_times = []
    per_image_rows = []
    for test_dict in test_loader:
        image, labels, fname = test_dict['images'].to(device), test_dict['labels'], test_dict['fname']
        with torch.no_grad():
            if torch.cuda.is_available() and str(device).startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pyramid2 = flow(image) 
            pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid2, feat_mean)]
            diff = flow.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True) # b,1,h,w

            if torch.cuda.is_available() and str(device).startswith('cuda'):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            diff_list.append(diff.cpu())
            labels_list.append(labels.cpu() == 1)
            inference_times.append(elapsed_ms)

            diff_np = np.squeeze(diff.cpu().numpy())
            amap_norm = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8)
            np.savez(f'{mpfm.test_result}/{fname[0]}.npz', amap_norm=amap_norm)

            per_image_rows.append({
                'fname': fname[0],
                'image_label': int((labels.cpu() == 1).amax().item()),
                'image_score': float(diff.cpu().amax().item()),
                'image_score_norm': 0.0,
                'inference_time_ms': elapsed_ms,
            })

            # import matplotlib.pyplot as plt
            # plt.imsave(f'{mpfm.test_result}/{fname[0]}.png', amap_norm, cmap='jet')

    labels_all = torch.concat(labels_list, dim=0)[:, 0]
    amaps = torch.concat(diff_list, dim=0)[:, 0]

    pixel_auroc = safe_roc_auc(labels_all.flatten(), amaps.flatten())
    image_labels = labels_all.amax((-1, -2))
    image_scores = amaps.amax((-1, -2))
    image_auroc = safe_roc_auc(image_labels, image_scores)
    pixel_pro = float(compute_pro_score_fast(amaps, labels_all)) if labels_all.any() else float('nan')

    amaps_norm = normalize_tensor(amaps)
    image_scores_norm = normalize_tensor(image_scores)

    if threshold is None and trained_pixel_threshold is not None:
        pixel_threshold = float(trained_pixel_threshold)
        pixel_pred = amaps > pixel_threshold
        pixel_threshold_mode = 'raw_trained'
    elif threshold is None:
        pixel_threshold = 0.5
        pixel_pred = amaps_norm > pixel_threshold
        pixel_threshold_mode = 'normalized_fallback'
    else:
        pixel_threshold = float(threshold)
        pixel_pred = amaps > pixel_threshold
        pixel_threshold_mode = 'raw_manual'

    if image_threshold is None and trained_image_threshold is not None:
        image_threshold_value = float(trained_image_threshold)
        image_pred = image_scores > image_threshold_value
        image_threshold_mode = 'raw_trained'
    elif image_threshold is None:
        image_threshold_value = pixel_threshold
        if pixel_threshold_mode.startswith('raw'):
            image_pred = image_scores > image_threshold_value
            image_threshold_mode = pixel_threshold_mode
        else:
            image_pred = image_scores_norm > image_threshold_value
            image_threshold_mode = 'normalized_fallback'
    else:
        image_threshold_value = float(image_threshold)
        image_pred = image_scores > image_threshold_value
        image_threshold_mode = 'raw_manual'

    pixel_tn, pixel_fp, pixel_fn, pixel_tp = confusion_counts(labels_all, pixel_pred)
    image_tn, image_fp, image_fn, image_tp = confusion_counts(image_labels, image_pred)

    for row, score_norm in zip(per_image_rows, image_scores_norm.tolist()):
        row['image_score_norm'] = float(score_norm)

    total_inference_time = float(sum(inference_times))
    avg_inference_time = total_inference_time / max(len(inference_times), 1)
    fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else float('nan')

    metrics = {
        'cls_name': cls_name,
        'test_datapath': test_datapath,
        'num_images': len(test_dataset),
        'pixel_auroc': pixel_auroc,
        'image_auroc': image_auroc,
        'pixel_pro': pixel_pro,
        'pixel_threshold': pixel_threshold,
        'image_threshold': image_threshold_value,
        'pixel_threshold_mode': pixel_threshold_mode,
        'image_threshold_mode': image_threshold_mode,
        'threshold_source': threshold_source,
        'pixel_tn': pixel_tn,
        'pixel_fp': pixel_fp,
        'pixel_fn': pixel_fn,
        'pixel_tp': pixel_tp,
        'image_tn': image_tn,
        'image_fp': image_fp,
        'image_fn': image_fn,
        'image_tp': image_tp,
        'total_inference_time_ms': total_inference_time,
        'avg_inference_time_ms': avg_inference_time,
        'fps': fps,
    }

    metrics_path, per_image_path, confusion_path = write_test_csvs(mpfm, metrics, per_image_rows)

    print(f"   Pixel-AUROC: {pixel_auroc}")
    print(f"   Image-AUROC: {image_auroc}")
    print(f"   Pixel-PRO: {pixel_pro}")
    print(f"   Pixel threshold: {pixel_threshold} ({pixel_threshold_mode})")
    print(f"   Image threshold: {image_threshold_value} ({image_threshold_mode})")
    print(f"   Pixel confusion matrix [tn, fp, fn, tp]: {[pixel_tn, pixel_fp, pixel_fn, pixel_tp]}")
    print(f"   Image confusion matrix [tn, fp, fn, tp]: {[image_tn, image_fp, image_fn, image_tp]}")
    print(f"   Avg inference time: {avg_inference_time} ms")
    print(f"   Metrics CSV: {metrics_path}")
    print(f"   Per-image CSV: {per_image_path}")
    print(f"   Confusion matrix CSV: {confusion_path}")
            
    message = {
        "status": "complete",
        "message": {
            "pixel_auroc": pixel_auroc,
            "image_auroc": image_auroc,
            "pixel_pro": pixel_pro,
            "pixel_threshold": pixel_threshold,
            "image_threshold": image_threshold_value,
            "avg_inference_time_ms": avg_inference_time,
            "metrics_csv": metrics_path,
            "per_image_csv": per_image_path,
            "confusion_matrix_csv": confusion_path,
        }
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)
    model_data.close()
    return metrics

def loadfm(device, mpfm, model_data=None):
    data = None
    if model_data is not None and 'feat_mean' in model_data:
        feat_mean_arr = model_data['feat_mean']
    else:
        data = np.load(f'{mpfm.weight_path}/feat_mean.npz', allow_pickle=True)
        feat_mean_arr = data['feat_mean']

    feat_mean_tensor = []
    try:
        for arr in feat_mean_arr:
            if isinstance(arr, np.ndarray):
                tensor = torch.from_numpy(arr).to(device)
            else:
                tensor = torch.tensor(arr).to(device)
            feat_mean_tensor.append(tensor)
    finally:
        if data is not None:
            data.close()

    return feat_mean_tensor

if __name__ == '__main__':
    base_mpfm, jobs = resolve_test_jobs()
    summary_rows = []

    for job in jobs:
        run_mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, job['version'])
        run_hyp = load_hyp_for_mpfm(run_mpfm)
        vn_dims = get_vn_dims(run_hyp.volumeNorm)
        resnetX = 0 if run_hyp.encoder=='none' else int(run_hyp.encoder[6:])

        category = job['category'] if job['category'] is not None else job['version'].replace('\\', '/').split('/')[-1]
        print(f"[category] {category} -> version {job['version']}")

        metrics = test(
            run_hyp.encoder,
            run_hyp.numLayer,
            vn_dims,
            ksize=run_hyp.kernelSize,
            channel=run_hyp.numChannel,
            num_stack=run_hyp.numStack,
            device=f'cuda:{run_hyp.gpu}',
            mpfm=run_mpfm,
            dataset_name=job['dataset'],
            category_name=job['category'],
            dataset_root=opt.dataset_root,
            threshold=opt.threshold,
            image_threshold=opt.image_threshold,
        )
        metrics['version'] = job['version']
        metrics['category'] = category
        summary_rows.append(metrics)

    if len(summary_rows) > 1:
        summary_path = write_summary_csv(base_mpfm, summary_rows)
        print(f"Summary CSV: {summary_path}")
