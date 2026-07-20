#!/usr/bin/env python3
"""
실험 결과 로그에서 metrics/mAP50-95(B) 기준 최고 성능 행을 추출하여 하나의 CSV로 저장하는 스크립트
"""

import csv
import os
from pathlib import Path

def extract_best_results():
    """
    runs/detect 경로의 모든 실험 결과에서 metrics/mAP50-95(B) 기준 최고 행을 추출
    """
    # 경로 설정
    base_dir = Path("runs/detect")
    output_file = "best_results_summary.csv"
    
    if not base_dir.exists():
        print(f"오류: {base_dir} 경로가 존재하지 않습니다.")
        return
    
    # 결과를 저장할 리스트
    best_rows = []
    experiment_names = []
    headers = None
    
    # runs/detect 하위의 모든 디렉토리 순회
    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        results_csv = exp_dir / "results.csv"
        
        # results.csv 파일이 존재하는지 확인
        if not results_csv.exists():
            print(f"경고: {results_csv} 파일이 없습니다. 건너뜁니다.")
            continue
        
        try:
            # CSV 파일 읽기
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                print(f"경고: {results_csv} 파일이 비어있습니다. 건너뜁니다.")
                continue
            
            # 헤더 저장 (첫 번째 파일에서)
            if headers is None:
                headers = list(rows[0].keys())
            
            # metrics/mAP50-95(B) 컬럼이 존재하는지 확인
            if "metrics/mAP50-95(B)" not in rows[0]:
                print(f"경고: {results_csv}에 'metrics/mAP50-95(B)' 컬럼이 없습니다. 건너뜁니다.")
                continue
            
            # metrics/mAP50-95(B) 값이 최대인 행 찾기
            max_value = float(rows[0]["metrics/mAP50-95(B)"])
            best_row = rows[0]
            
            for row in rows[1:]:
                try:
                    value = float(row["metrics/mAP50-95(B)"])
                    if value > max_value:
                        max_value = value
                        best_row = row
                except (ValueError, KeyError):
                    continue
            
            # 결과 저장
            best_rows.append(best_row)
            experiment_names.append(exp_dir.name)
            
            epoch = best_row.get('epoch', 'N/A')
            print(f"✓ {exp_dir.name}: 최고 mAP50-95(B) = {max_value:.5f} (epoch {epoch})")
            
        except Exception as e:
            print(f"오류: {results_csv} 처리 중 오류 발생: {e}")
            continue
    
    if not best_rows:
        print("추출된 결과가 없습니다.")
        return
    
    # CSV 파일로 저장 (인덱스 컬럼 포함)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if headers:
            # experiment_name을 첫 번째 컬럼으로 추가
            fieldnames = ['experiment_name'] + headers
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 최고 mAP50-95(B) 값 찾기
            max_map_value = 0.0
            best_experiment = ""
            
            for exp_name, row in zip(experiment_names, best_rows):
                row_with_index = {'experiment_name': exp_name}
                row_with_index.update(row)
                writer.writerow(row_with_index)
                
                try:
                    map_value = float(row.get("metrics/mAP50-95(B)", 0))
                    if map_value > max_map_value:
                        max_map_value = map_value
                        best_experiment = exp_name
                except (ValueError, TypeError):
                    pass
    
    print(f"\n✓ 총 {len(best_rows)}개의 실험 결과를 {output_file}에 저장했습니다.")
    if best_experiment:
        print(f"  최고 mAP50-95(B) 값: {max_map_value:.5f}")
        print(f"  최고 성능 실험: {best_experiment}")

if __name__ == "__main__":
    extract_best_results()
