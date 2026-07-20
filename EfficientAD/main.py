#!/usr/bin/python
# -*- coding: utf-8 -*-
"""EfficientAD continuity 실험 드라이버 (단일 그리드).

맨 위 리스트(ALPHAS / BETAS / DROPOUT_RATIOS)로 조합을 만들어 efficientad.py를
돌리고 metrics_summary.csv를 모아 표로 보여준다. `--cont_loss_type`이
**어떤 하이퍼파라미터를 스윕할지** 결정한다:

  next      → DROPOUT_RATIOS 만 스윕         (feature dropout)
  diff      → ALPHAS, BETAS 만 스윕          (1·2차 차분 가중치)
  next+diff → DROPOUT_RATIOS × ALPHAS × BETAS (전부)

`--grid_mode` 는 diff/next+diff에서 alpha/beta 짝을 정한다:
  paired → alpha=beta 만 / full → 모든 alpha×beta 조합.

예) 기본 리스트 + next+diff:  paired = 15 run, full = 75 run.

baseline(continuity_test 0 = 원본)은 학습시간 절약을 위해 기본 재사용한다.
처음 한 번만 `--run_baseline` 로 생성하면 이후 Δ 계산에 재활용된다.

예:
  python main.py --cont_loss_type next  -s all              # mask 스윕
  python main.py --cont_loss_type next+diff -s screw        # 전체 그리드
  python main.py --cont_loss_type next  -s all --run_baseline   # baseline 1회 생성
  python main.py --cont_loss_type next  -s all --analyze_only   # 표만
"""
import argparse
import csv
import glob
import itertools
import os
import subprocess
import sys

# === 스윕 대상 값들 (여기서 조합을 만든다) ===
ALPHAS = [1]
BETAS = [1]
DROPOUT_RATIOS = [0.1, 0.2, 0.3]
##############################################
SUB_DATASET = ['carpet']
CONT_LOSS_TYPE = 'diff' # 스윕 대상 결정: next=dropout / diff=alpha,beta / next+diff=모두
##############################################

def get_argparse():
    p = argparse.ArgumentParser(description='EfficientAD continuity 실험 드라이버')
    # 실행 대상
    p.add_argument('--dataset', default='mvtec_ad',
                   choices=['mvtec_ad', 'mvtec_loco'])
    p.add_argument('-s', '--subdatasets', nargs='+', default=SUB_DATASET,
                   help="카테고리 목록 또는 'all'")
    p.add_argument('-o', '--output_dir', default='results/grid_search')
    p.add_argument('-m', '--model_size', default='small',
                   choices=['small', 'medium'])
    p.add_argument('-w', '--weights', default='models/teacher_small.pth')
    p.add_argument('-i', '--imagenet_train_path', default='none')
    p.add_argument('-t', '--train_steps', type=int, default=10000)
    p.add_argument('--mvtec_ad_path',
                   default='E:/0_dataset/mvtec_anomaly_detection')
    p.add_argument('--mvtec_loco_path',
                   default='E:/0_dataset/mvtec_loco_anomaly_detection')
    p.add_argument('--target_fpr', type=float, default=0.001)
    p.add_argument('--npp_mode', default='l1', choices=['l1', 'l2'])
    # continuity 스윕 설정
    p.add_argument('--cont_loss_type', default=CONT_LOSS_TYPE,
                   choices=['next', 'diff', 'next+diff'],
                   help='스윕 대상 결정: next=dropout / diff=alpha,beta / next+diff=모두')
    p.add_argument('--grid_mode', default='paired', choices=['paired', 'full'],
                   help='diff/next+diff에서 paired: alpha=beta / full: alpha×beta')
    p.add_argument('--dropout_ratios', nargs='+', type=float, default=DROPOUT_RATIOS,
                   help='next/next+diff에서 스윕할 dropout 값들 (기본 DROPOUT_RATIOS)')
    p.add_argument('--alphas', nargs='+', type=float, default=ALPHAS,
                   help='diff/next+diff에서 스윕할 alpha 값들 (기본 ALPHAS)')
    p.add_argument('--betas', nargs='+', type=float, default=BETAS,
                   help='diff/next+diff(full)에서 스윕할 beta 값들 (기본 BETAS)')
    p.add_argument('--cont_train_w', type=float, default=1.0,
                   help='학습 손실에 continuity 비중')
    p.add_argument('--cont_score_w', type=float, default=0.0,
                   help='추론 점수의 map_cont 비중 (0=끔; map_cont는 무용으로 확인됨)')
    p.add_argument('--run_baseline', action='store_true',
                   help='baseline(test=0)도 학습. 기본은 기존 baseline 재사용')
    # 실행 제어
    p.add_argument('--skip_existing', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--analyze_only', action='store_true')
    p.add_argument('--stop_on_error', action='store_true')
    return p.parse_args()


# ---------------- 공용 유틸 ----------------
def find_run_dir(base):
    hits = sorted(glob.glob(os.path.join(base, '**', 'metrics_summary.csv'),
                            recursive=True))
    return os.path.dirname(hits[-1]) if hits else None


def read_aurocs(run_dir):
    out = {}
    if not run_dir:
        return out
    path = os.path.join(run_dir, 'metrics_summary.csv')
    if not os.path.isfile(path):
        return out
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['subdataset']] = float(row['image_auroc'])
    return out


def base_command(config, subdataset, out, continuity_test):
    return [sys.executable, 'efficientad.py',
            '--dataset', config.dataset, '-s', subdataset, '-o', out,
            '--model_size', config.model_size, '--weights', config.weights,
            '--imagenet_train_path', config.imagenet_train_path,
            '--train_steps', str(config.train_steps),
            '--target_fpr', str(config.target_fpr),
            '--npp_mode', config.npp_mode,
            '--mvtec_ad_path', config.mvtec_ad_path,
            '--mvtec_loco_path', config.mvtec_loco_path,
            '--continuity_test', str(continuity_test)]


def run_one(config, out, cmd):
    if config.skip_existing and find_run_dir(out):
        print('[SKIP] {}'.format(out))
        return
    print('[RUN] {}'.format(' '.join(cmd)))
    if config.dry_run:
        return
    os.makedirs(out, exist_ok=True)
    rc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__))).returncode
    if rc and config.stop_on_error:
        raise SystemExit(rc)


# ---------------- 조합 생성 ----------------
def build_combos(config):
    """(dropout, alpha, beta) 조합 목록. loss_type에 따라 쓰는 축만 스윕.

    스윕 값은 CLI(--dropout_ratios/--alphas/--betas)로 덮어쓸 수 있고,
    기본은 모듈 상단 리스트.
    """
    loss_type = config.cont_loss_type
    use_dropout = loss_type in ('next', 'next+diff')
    use_diff = loss_type in ('diff', 'next+diff')
    dropouts = config.dropout_ratios if use_dropout else [0.0]
    if use_diff:
        ab = ([(a, a) for a in config.alphas] if config.grid_mode == 'paired'
              else list(itertools.product(config.alphas, config.betas)))
    else:
        ab = [(0.0, 0.0)]
    return [(d, a, b) for d in dropouts for (a, b) in ab]


def combo_name(loss_type, d, a, b):
    parts = []
    if loss_type in ('diff', 'next+diff'):
        parts += ['alpha{:.2f}'.format(a), 'beta{:.2f}'.format(b)]
    if loss_type in ('next', 'next+diff'):
        parts += ['dropout{:.1f}'.format(d)]
    return '_'.join(parts).replace('.', 'p')


def combo_args(config, d, a, b):
    extra = ['--cont_loss_type', config.cont_loss_type,
             '--cont_train_w', str(config.cont_train_w),
             '--cont_score_w', str(config.cont_score_w)]
    if config.cont_loss_type in ('next', 'next+diff'):
        extra += ['--cont_dropout', str(d)]
    if config.cont_loss_type in ('diff', 'next+diff'):
        extra += ['--cont_alpha', str(a), '--cont_beta', str(b)]
    return extra


# ---------------- 실행 + 집계 ----------------
def run_all(config, combos):
    base = config.output_dir
    if config.run_baseline and not config.analyze_only:
        for sub in config.subdatasets:
            out = os.path.join(base, sub, 'baseline')
            run_one(config, out, base_command(config, sub, out, 0))

    results = {}
    for sub in config.subdatasets:
        for d, a, b in combos:
            name = combo_name(config.cont_loss_type, d, a, b)
            out = os.path.join(base, sub, name)
            if not config.analyze_only:
                cmd = base_command(config, sub, out, 1) + combo_args(config, d, a, b)
                run_one(config, out, cmd)
            results[(sub, name)] = read_aurocs(find_run_dir(out))

    baselines = {}
    for sub in config.subdatasets:
        baselines.update(read_aurocs(find_run_dir(os.path.join(base, sub, 'baseline'))))
    return results, baselines


def analyze(config, combos, results, baselines):
    # (category, combo_name) -> auroc 로 평탄화 ('all'이면 카테고리 15개로 펼쳐짐)
    flat = {}
    for (sub, name), aurocs in results.items():
        for cat, v in aurocs.items():
            flat[(cat, name)] = v
    cats = sorted(set(c for (c, _) in flat))
    if not cats:
        print('(결과 없음 — 학습을 먼저 돌리거나 경로 확인)')
        return

    bcats = [c for c in cats if c in baselines]
    bmean = sum(baselines[c] for c in bcats) / len(bcats) if bcats else None
    print('\n=== 결과: 카테고리 {}개 평균 | baseline = {} ==='.format(
        len(cats), '{:.3f}'.format(bmean) if bmean is not None else '없음'))
    print('%-26s %10s %9s %8s' % ('combo', 'meanAUROC', 'meanΔ', '양수'))
    for d, a, b in combos:
        name = combo_name(config.cont_loss_type, d, a, b)
        vals = [flat[(c, name)] for c in cats if (c, name) in flat]
        if not vals:
            print('%-26s %10s' % (name, '(미완료)'))
            continue
        mauroc = sum(vals) / len(vals)
        dd = [flat[(c, name)] - baselines[c]
              for c in cats if (c, name) in flat and c in baselines]
        if dd:
            print('%-26s %10.3f %+9.3f %5d/%d'
                  % (name, mauroc, sum(dd) / len(dd), sum(1 for x in dd if x > 0), len(dd)))
        else:
            print('%-26s %10.3f %9s %8s' % (name, mauroc, '-', '-'))


def main():
    config = get_argparse()
    combos = build_combos(config)
    print('조합 {}개 (cont_loss_type={}, grid_mode={}) × 카테고리 {}'.format(
        len(combos), config.cont_loss_type, config.grid_mode,
        ' '.join(config.subdatasets)))
    results, baselines = run_all(config, combos)
    analyze(config, combos, results, baselines)


if __name__ == '__main__':
    main()
