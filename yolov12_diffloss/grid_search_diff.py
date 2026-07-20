#!/usr/bin/env python3
"""
차분 로스(Diff Loss) 하이퍼파라미터 Grid Search 스크립트
alpha, beta에 대한 모든 조합을 체계적으로 테스트합니다.
"""

import itertools
import subprocess
import os
import sys
from pathlib import Path
import csv
from datetime import datetime

# 환경 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# Grid Search 파라미터 정의
alpha_values = [0.0, 0.1, 0.3, 0.5]  # 1차 차분 가중치 (alpha)
beta_values = [0.0, 0.5, 1.0, 1.5, 2.0]  # 2차 차분 가중치 (beta)
fpn_sources = ["14,17,20"]  # FPN_SOURCES (v12_diff에서는 14,17,20 사용)

# 결과 저장 디렉토리
results_dir = Path("runs/detect/grid_search_diff")
results_dir.mkdir(parents=True, exist_ok=True)
results_csv = results_dir / "grid_search_results.csv"

# 모든 조합 생성
all_combinations = list(itertools.product(
    alpha_values,
    beta_values,
    fpn_sources
))

# 필터링: alpha=0, beta=0인 경우 스킵 (차분 로스가 의미 없음)
filtered_combinations = []
for alpha, beta, fpn in all_combinations:
    if alpha == 0.0 and beta == 0.0:
        continue  # 둘 다 0이면 스킵
    filtered_combinations.append((alpha, beta, fpn))

print(f"==========================================")
print(f"차분 로스(Diff Loss) Grid Search 시작")
print(f"==========================================")
print(f"전체 조합 수: {len(all_combinations)}")
print(f"실행할 조합 수: {len(filtered_combinations)} (alpha=0, beta=0 제외)")
print(f"Alpha 값 (1차 차분 가중치): {alpha_values}")
print(f"Beta 값 (2차 차분 가중치): {beta_values}")
print(f"FPN Sources: {fpn_sources}")
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"==========================================")
print()

# CSV 헤더 작성
with open(results_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'iteration', 'diff_alpha', 'diff_beta', 
        'diff_fpn_sources', 'fitness', 'mAP50', 'mAP50-95', 'precision', 'recall',
        'status', 'save_dir'
    ])

# 각 조합 실행
for idx, (alpha, beta, fpn) in enumerate(filtered_combinations, 1):
    print(f"\n[{idx}/{len(filtered_combinations)}] 실행 중...")
    print(f"  Alpha: {alpha}, Beta: {beta}, FPN: {fpn}")
    
    # train.py 실행
    cmd = [
        "python", "train.py",
        f"--npp_alpha={alpha}",
        f"--npp_beta={beta}",
        f"--npp_fpn_sources={fpn}",
        "--batch=2",
        "--device=0",
        "--epochs=400"  # 빠른 테스트용, 필요시 변경
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            env=os.environ.copy(),
            # capture_output=False로 설정하여 학습 중 출력이 화면에 표시되도록 함
            check=True
        )
        
        # 결과 디렉토리 찾기
        fpn_str = fpn.replace(',', '_')
        save_dir_name = f"train_diff_alpha{alpha}_beta{beta}_fpn{fpn_str}"
        save_dir = Path(f"runs/detect/{save_dir_name}")
        
        # 체크포인트에서 메트릭 읽기
        metrics = {}
        ckpt_file = save_dir / "weights" / "best.pt"
        if not ckpt_file.exists():
            ckpt_file = save_dir / "weights" / "last.pt"
        
        if ckpt_file.exists():
            import torch
            ckpt = torch.load(ckpt_file, map_location='cpu')
            metrics = ckpt.get("train_metrics", {})
        
        fitness = metrics.get("fitness", 0.0)
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)
        
        status = "success"
        print(f"  ✓ 완료 - Fitness: {fitness:.5f}, mAP50: {map50:.5f}")
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 실패: {e}")
        fitness = 0.0
        map50 = 0.0
        map50_95 = 0.0
        precision = 0.0
        recall = 0.0
        status = "failed"
        save_dir_name = ""
    
    # 결과를 CSV에 저장
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            idx, alpha, beta, fpn,
            round(fitness, 5), round(map50, 5), round(map50_95, 5),
            round(precision, 5), round(recall, 5),
            status, save_dir_name
        ])

print(f"\n==========================================")
print(f"Grid Search 완료!")
print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"결과는 {results_csv}에 저장되었습니다.")
print(f"==========================================")

# 최고 결과 출력
if results_csv.exists():
    import pandas as pd
    try:
        df = pd.read_csv(results_csv)
        best_idx = df['fitness'].idxmax()
        best_row = df.iloc[best_idx]
        print(f"\n최고 결과:")
        print(f"  Alpha (1차 차분): {best_row['diff_alpha']}")
        print(f"  Beta (2차 차분): {best_row['diff_beta']}")
        print(f"  FPN Sources: {best_row['diff_fpn_sources']}")
        print(f"  Fitness: {best_row['fitness']:.5f}")
        print(f"  mAP50: {best_row['mAP50']:.5f}")
        print(f"  mAP50-95: {best_row['mAP50-95']:.5f}")
    except ImportError:
        print("\n최고 결과를 확인하려면 pandas를 설치하세요: pip install pandas")
    except Exception as e:
        print(f"\n최고 결과 확인 중 오류: {e}")
