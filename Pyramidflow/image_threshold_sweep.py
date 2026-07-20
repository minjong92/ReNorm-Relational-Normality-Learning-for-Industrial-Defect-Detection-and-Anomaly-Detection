"""
image_threshold_sweep.py — 저장된 추론 결과로 '이미지 단위' confusion matrix를
여러 임계값(trained / percentile / manual)에서 재계산한다.

[배경]
  test.py 가 카테고리마다 저장하는 산출물은 두 가지다.
    1) inference_result/{fname}.npz : per-image 로 0~1 정규화된 anomaly map(amap_norm)
    2) inference_result/per_image_results.csv : 이미지별 raw 점수(image_score) + 라벨

  '픽셀' confusion 은 raw diff 맵 + raw 임계값으로 계산되는데(test.py 참고),
  .npz 에는 per-image 정규화 맵만 있고 raw 맵은 저장되지 않는다.
  → 저장된 .npz 만으로는 '픽셀' confusion 을 충실히 복원할 수 없다.
     (픽셀 운영점을 바꾸려면 test.py 를 --threshold 로 다시 추론하는 게 정석)

  반면 '이미지' 점수(image_score)는 raw 값이 csv 에 그대로 남아 있으므로,
  '이미지' confusion 은 모델/데이터 재실행 없이 어떤 임계값으로도 다시 뽑을 수 있다.
  carpet/leather/toothbrush 의 TN=0(정상 전부 오탐) 문제는 정확히 이 '이미지' 단위 문제다.

[하는 일]
  results-root 아래 각 카테고리의 per_image_results.csv 를 읽어, 아래 임계값들에 대해
  이미지 confusion(tn/fp/fn/tp)과 recall/fpr/precision 을 계산하고 out 에 CSV 로 저장한다.
    - trained   : 학습 임계값(metrics.csv 의 image_threshold) 재현 → 기존 결과와 일치 확인용
    - p{XX}     : test-good 점수의 percentile 임계값 (예: 99, 99.5)
    - manual    : 사용자가 준 raw 임계값(옵션)
  이미지 AUROC(임계값 무관)도 참고로 같이 출력한다.

[주의]
  percentile 임계값은 test-good 점수로 계산하므로 '진단용'이다(test 기반 = leakage).
  실제 보고/운영 임계값은 검증(val-good)에서 정해야 한다.
"""

import os
import csv
import argparse
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


DEFAULT_RESULTS_ROOT = (
    r"D:\py_project\MOAI_anomaly_for_continuity_loss"
    r"\moai\test_project\test_subproject\test_task\--loss0"
)
DEFAULT_OUT = (
    r"D:\py_project\MOAI_anomaly_for_continuity_loss"
    r"\moai\test_project\test_subproject\test_task\temp"
)


def find_categories(results_root):
    """per_image_results.csv 를 가진 하위 폴더(=카테고리)만 골라낸다."""
    cats = []
    for name in sorted(os.listdir(results_root)):
        per_image = os.path.join(results_root, name, 'inference_result', 'per_image_results.csv')
        if os.path.isfile(per_image):
            cats.append(name)
    return cats


def load_per_image(per_image_path):
    labels, scores = [], []
    with open(per_image_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(float(row['image_label'])))
            scores.append(float(row['image_score']))
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def load_trained_image_threshold(category_dir):
    """metrics.csv 에서 학습 당시 image_threshold 를 읽는다(없으면 None)."""
    metrics_path = os.path.join(category_dir, 'inference_result', 'metrics.csv')
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get('image_threshold')
            if value not in (None, ''):
                try:
                    return float(value)
                except ValueError:
                    return None
    return None


def image_confusion(labels, scores, thr):
    """test.py 와 동일하게 'score > thr' 를 양성(anomaly)으로 본다."""
    pred = scores > thr
    pos = labels == 1
    tp = int(np.sum(pred & pos))
    fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    fpr = fp / (fp + tn) if (fp + tn) else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, recall=recall, fpr=fpr, precision=precision)


def image_auroc(labels, scores):
    if not _HAS_SKLEARN or np.unique(labels).size < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))


def build_rows(category, labels, scores, trained_thr, percentiles, manual_thrs):
    good = scores[labels == 0]
    num_good = int(np.sum(labels == 0))
    num_defect = int(np.sum(labels == 1))
    auroc = image_auroc(labels, scores)

    jobs = []  # (mode, threshold)
    if trained_thr is not None:
        jobs.append(('trained', float(trained_thr)))
    for p in percentiles:
        if good.size == 0:
            continue
        jobs.append((f'test_good_p{p:g}', float(np.percentile(good, p))))
    for m in manual_thrs:
        jobs.append((f'manual_{m:g}', float(m)))

    rows = []
    for mode, thr in jobs:
        cm = image_confusion(labels, scores, thr)
        rows.append({
            'category': category,
            'num_good': num_good,
            'num_defect': num_defect,
            'image_auroc': auroc,
            'threshold_mode': mode,
            'threshold': thr,
            'image_tn': cm['tn'],
            'image_fp': cm['fp'],
            'image_fn': cm['fn'],
            'image_tp': cm['tp'],
            'recall': cm['recall'],
            'fpr': cm['fpr'],
            'precision': cm['precision'],
        })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_by_mode(all_rows):
    modes = []
    for r in all_rows:
        if r['threshold_mode'] not in modes:
            modes.append(r['threshold_mode'])
    out = []
    for mode in modes:
        sub = [r for r in all_rows if r['threshold_mode'] == mode]

        def nanmean(key):
            vals = [r[key] for r in sub if not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float('nan')

        out.append({
            'threshold_mode': mode,
            'num_categories': len(sub),
            'mean_recall': nanmean('recall'),
            'mean_fpr': nanmean('fpr'),
            'mean_precision': nanmean('precision'),
            'mean_image_auroc': nanmean('image_auroc'),
            'total_image_fn': int(sum(r['image_fn'] for r in sub)),
            'total_image_fp': int(sum(r['image_fp'] for r in sub)),
        })
    return out


def parse_floats(text):
    if not text:
        return []
    return [float(x) for x in text.split(',') if x.strip() != '']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-root', default=DEFAULT_RESULTS_ROOT,
                        help='카테고리 폴더들이 들어있는 결과 루트')
    parser.add_argument('--out', default=DEFAULT_OUT, help='결과 CSV 저장 디렉터리')
    parser.add_argument('--percentiles', default='95,99,99.5',
                        help='test-good 점수 기준 임계값 percentile 목록 (예: "95,99,99.5")')
    parser.add_argument('--manual', default='',
                        help='수동 raw 이미지 임계값 목록 (예: "0.35,0.4")')
    opt = parser.parse_args()

    percentiles = parse_floats(opt.percentiles)
    manual_thrs = parse_floats(opt.manual)

    if not os.path.isdir(opt.results_root):
        raise FileNotFoundError(f"results-root not found: {opt.results_root}")
    os.makedirs(opt.out, exist_ok=True)

    categories = find_categories(opt.results_root)
    if not categories:
        raise FileNotFoundError(
            f"No per_image_results.csv found under: {opt.results_root}"
        )

    print(f"[results-root] {opt.results_root}")
    print(f"[out]          {opt.out}")
    print(f"[categories]   {len(categories)}: {', '.join(categories)}")
    print(f"[percentiles]  {percentiles}")
    if manual_thrs:
        print(f"[manual thr]   {manual_thrs}")
    print(f"[note] 픽셀 confusion 은 저장된 .npz(정규화 맵)로 복원 불가 → 이미지 단위만 재계산함\n")

    all_rows = []
    for cat in categories:
        cat_dir = os.path.join(opt.results_root, cat)
        labels, scores = load_per_image(
            os.path.join(cat_dir, 'inference_result', 'per_image_results.csv')
        )
        trained_thr = load_trained_image_threshold(cat_dir)
        rows = build_rows(cat, labels, scores, trained_thr, percentiles, manual_thrs)
        all_rows.extend(rows)

        # 콘솔 요약
        print(f"== {cat}  (good={int(np.sum(labels==0))}, defect={int(np.sum(labels==1))}, "
              f"image-AUROC={rows[0]['image_auroc']:.4f})")
        for r in rows:
            print(f"   {r['threshold_mode']:>16}  thr={r['threshold']:.4f}  "
                  f"[tn={r['image_tn']:>3} fp={r['image_fp']:>3} fn={r['image_fn']:>3} tp={r['image_tp']:>3}]  "
                  f"recall={r['recall']:.3f}  fpr={r['fpr']:.3f}  prec={r['precision']:.3f}")
        print()

    detail_fields = ['category', 'num_good', 'num_defect', 'image_auroc',
                     'threshold_mode', 'threshold',
                     'image_tn', 'image_fp', 'image_fn', 'image_tp',
                     'recall', 'fpr', 'precision']
    detail_path = os.path.join(opt.out, 'image_confusion_recompute.csv')
    write_csv(detail_path, all_rows, detail_fields)

    mean_rows = mean_by_mode(all_rows)
    mean_fields = ['threshold_mode', 'num_categories', 'mean_recall', 'mean_fpr',
                   'mean_precision', 'mean_image_auroc', 'total_image_fn', 'total_image_fp']
    mean_path = os.path.join(opt.out, 'image_confusion_mean_by_mode.csv')
    write_csv(mean_path, mean_rows, mean_fields)

    print("[mode 별 평균]")
    for m in mean_rows:
        print(f"   {m['threshold_mode']:>16}  mean_recall={m['mean_recall']:.3f}  "
              f"mean_fpr={m['mean_fpr']:.3f}  mean_prec={m['mean_precision']:.3f}  "
              f"FN합={m['total_image_fn']}  FP합={m['total_image_fp']}")

    print(f"\n저장 완료:\n  {detail_path}\n  {mean_path}")


if __name__ == '__main__':
    main()
