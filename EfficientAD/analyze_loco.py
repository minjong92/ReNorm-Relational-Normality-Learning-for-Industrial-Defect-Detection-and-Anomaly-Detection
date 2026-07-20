#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MVTec LOCO 결과를 결함 유형별로 분리해 image AUROC를 계산한다.

continuity loss가 'logical 이상'에 특히 도움이 되는지 보려면 전체 AUROC가 아니라
good vs logical / good vs structural 을 따로 봐야 한다. efficientad.py 와
evaluate_existing_run.py 가 남긴 image_scores.csv(컬럼: defect_class, score)를 읽어
세 가지 AUROC를 산출한다.

학습 직후 결과(metrics/)와 비중 스윕 결과(eval_w*/metrics/)를 모두 자동으로 모아
열로 나란히 비교한다.

사용:
    python analyze_loco.py --run_dir results/mvtec_loco_anomaly_detection/<타임스탬프>
"""
import argparse
import csv
import glob
import os


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True,
                        help='LOCO run 디렉터리 (metrics/ 및/또는 eval_w*/ 포함).')
    return parser.parse_args()


def read_scores_by_class(scores_csv):
    by_class = {}
    with open(scores_csv, newline='', encoding='utf-8') as csv_file:
        for row in csv.DictReader(csv_file):
            by_class.setdefault(row['defect_class'], []).append(float(row['score']))
    return by_class


def auroc(good_scores, anomaly_scores):
    """순위 기반 AUROC (Mann-Whitney U, 동점은 평균 순위). sklearn 불필요."""
    n_pos = len(anomaly_scores)
    n_neg = len(good_scores)
    if n_pos == 0 or n_neg == 0:
        return None
    items = ([(s, 1) for s in anomaly_scores] + [(s, 0) for s in good_scores])
    items.sort(key=lambda x: x[0])
    ranks = [0.0] * len(items)
    i = 0
    while i < len(items):
        j = i
        while j < len(items) and items[j][0] == items[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based 순위 i+1..j 의 평균
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_ranks_pos = sum(r for r, (_, lbl) in zip(ranks, items) if lbl == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc * 100.0


def subdataset_aurocs(scores_csv):
    by_class = read_scores_by_class(scores_csv)
    good = by_class.get('good', [])
    logical = by_class.get('logical_anomalies', [])
    structural = by_class.get('structural_anomalies', [])
    all_anomalies = [s for cls, scores in by_class.items()
                     if cls != 'good' for s in scores]
    return {
        'logical': auroc(good, logical),
        'structural': auroc(good, structural),
        'overall': auroc(good, all_anomalies),
    }


def find_sources(run_dir):
    """(라벨, metrics_dir) 목록: 학습 직후 metrics/ + eval_w*/metrics/."""
    sources = []
    base_metrics = os.path.join(run_dir, 'metrics')
    if os.path.isdir(base_metrics):
        sources.append(('train', base_metrics))
    for eval_dir in sorted(glob.glob(os.path.join(run_dir, 'eval_w*'))):
        metrics_dir = os.path.join(eval_dir, 'metrics')
        if os.path.isdir(metrics_dir):
            sources.append((os.path.basename(eval_dir), metrics_dir))
    return sources


def collect(metrics_dir):
    results = {}
    for subdataset in sorted(os.listdir(metrics_dir)):
        scores_csv = os.path.join(metrics_dir, subdataset, 'image_scores.csv')
        if os.path.isfile(scores_csv):
            results[subdataset] = subdataset_aurocs(scores_csv)
    return results


def fmt(value):
    return '{:.2f}'.format(value) if value is not None else '-'


def print_table(title, results, sources, subdatasets, key):
    print('\n=== {} (image AUROC) ==='.format(title))
    labels = [label for label, _ in sources]
    print('\t'.join(['subdataset'] + labels))
    sums = {label: [] for label in labels}
    for subdataset in subdatasets:
        cells = [subdataset]
        for label in labels:
            value = results[label].get(subdataset, {}).get(key)
            cells.append(fmt(value))
            if value is not None:
                sums[label].append(value)
        print('\t'.join(cells))
    mean_cells = ['평균']
    for label in labels:
        vals = sums[label]
        mean_cells.append('{:.2f}'.format(sum(vals) / len(vals)) if vals else '-')
    print('\t'.join(mean_cells))


def main():
    config = get_argparse()
    sources = find_sources(config.run_dir)
    if not sources:
        raise SystemExit('metrics를 찾을 수 없습니다: {}'.format(config.run_dir))
    results = {label: collect(metrics_dir) for label, metrics_dir in sources}
    subdatasets = sorted({s for r in results.values() for s in r})
    print('Run: {}'.format(config.run_dir))
    print('Sources: {}'.format(', '.join(label for label, _ in sources)))
    for title, key in [('Logical anomalies', 'logical'),
                       ('Structural anomalies', 'structural'),
                       ('Overall (good vs all)', 'overall')]:
        print_table(title, results, sources, subdatasets, key)


if __name__ == '__main__':
    main()
