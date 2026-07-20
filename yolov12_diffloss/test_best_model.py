#!/usr/bin/env python3
"""
best_results_summary.csv의 모든 run에 대해
test 데이터셋에 대해 inference를 수행하고 결과를 CSV로 저장하는 스크립트
"""

import csv
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import os
import sys


def load_all_runs():
    """best_results_summary.csv에서 모든 run 정보 로드"""
    csv_path = Path("best_results_summary.csv")

    if not csv_path.exists():
        print(f"오류: {csv_path} 파일이 존재하지 않습니다.")
        return None

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    if "experiment_name" not in df.columns:
        print("오류: 'experiment_name' 컬럼이 없습니다.")
        print(f"사용 가능한 컬럼: {list(df.columns)}")
        return None

    print(f"총 {len(df)}개의 run을 찾았습니다.")
    
    # 최고 성능 모델 정보 출력 (참고용)
    if "metrics/mAP50-95(B)" in df.columns:
        best_idx = df["metrics/mAP50-95(B)"].idxmax()
        best_row = df.iloc[best_idx]
        print(f"\n참고: 최고 성능 모델 (validation 기준):")
        print(f"  Experiment: {best_row['experiment_name']}")
        print(f"  mAP50-95: {best_row['metrics/mAP50-95(B)']:.5f}")

    return df



def find_weight_file(experiment_name):
    """실험 이름으로부터 best.pt 파일 경로 찾기"""
    weight_path = Path(f"runs/detect/{experiment_name}/weights/best.pt")
    
    if not weight_path.exists():
        # last.pt로 시도
        weight_path = Path(f"runs/detect/{experiment_name}/weights/last.pt")
        if not weight_path.exists():
            print(f"오류: 가중치 파일을 찾을 수 없습니다: {experiment_name}")
            return None
        print(f"경고: best.pt가 없어서 last.pt를 사용합니다.")
    
    print(f"가중치 파일: {weight_path}")
    return weight_path

def evaluate_on_test(weight_path, experiment_name=None, data_yaml_path="data/data.yaml"):
    """test 데이터셋에 대해 모델 평가 수행"""
    if experiment_name:
        print(f"\n[{experiment_name}] Test 데이터셋 평가 시작")
    else:
        print("\n" + "="*50)
        print("Test 데이터셋 평가 시작")
        print("="*50)
    
    # 모델 로드
    print(f"모델 로드 중: {weight_path}")
    model = YOLO(str(weight_path))
    
    # Test 데이터셋 경로 확인
    test_images_path = Path(r"E:\mj\MODEL\contiunity_test\yolov12_test\v12_diff\dataset\images\test")
    test_labels_path = Path(r"E:\mj\MODEL\contiunity_test\yolov12_test\v12_diff\dataset\labels\test")
    
    if not test_images_path.exists():
        print(f"오류: Test 이미지 경로가 존재하지 않습니다: {test_images_path}")
        return None, None
    
    if not test_labels_path.exists():
        print(f"경고: Test 라벨 경로가 존재하지 않습니다: {test_labels_path}")
    
    # data.yaml 읽어서 test 경로 추가
    data_yaml = Path(data_yaml_path)
    if not data_yaml.exists():
        print(f"오류: data.yaml 파일이 존재하지 않습니다: {data_yaml_path}")
        return None, None
    
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # test 경로 추가 (YOLO가 test split을 인식하도록)
    data_config['test'] = 'images/test'
    
    # 임시 data.yaml 생성
    temp_data_yaml = Path("data/data_test.yaml")
    with open(temp_data_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    data_yaml_path = str(temp_data_yaml)
    print(f"Test 데이터셋 설정 파일 생성: {data_yaml_path}")
    print(f"  Test 이미지 경로: {test_images_path}")
    print(f"  Test 라벨 경로: {test_labels_path}")
    print(f"  data.yaml 내용:")
    print(f"    path: {data_config.get('path', 'N/A')}")
    print(f"    test: {data_config.get('test', 'N/A')}")
    
    # Validation 수행 (test 데이터셋에 대해, split='test'로 명시)
    # 각 run마다 고유한 이름으로 결과 저장
    eval_name = f"test_evaluation_{experiment_name}" if experiment_name else "test_evaluation"
    # 파일명에 사용할 수 없는 문자 제거
    eval_name = eval_name.replace("/", "_").replace("\\", "_")
    
    print("\nTest 데이터셋에 대해 평가 수행 중...")
    print("  주의: validation이 아닌 test 데이터셋을 사용합니다.")
    results = model.val(
        data=data_yaml_path,
        split='test',  # test split 명시적으로 지정
        project="runs/detect",
        name=eval_name,
        exist_ok=True,
        save_json=True,
        save_hybrid=False,
        plots=False  # 모든 run에 대해 plot 생성하면 용량이 너무 커질 수 있음
    )
    
    # results.csv 파일에서 메트릭 읽기
    results_csv_path = Path(f"runs/detect/{eval_name}/results.csv")
    metrics = {}
    
    if results_csv_path.exists():
        # CSV 파일의 마지막 행 읽기 (최종 결과)
        df_results = pd.read_csv(results_csv_path)
        if not df_results.empty:
            # 마지막 행이 최종 결과
            last_row = df_results.iloc[-1]
            print(f"CSV 컬럼명: {list(df_results.columns)}")
            
            # 컬럼명 매핑 (다양한 형식 지원)
            for col in df_results.columns:
                col_lower = col.lower()
                # precision 찾기
                if 'precision' in col_lower and '(b)' in col_lower:
                    metrics['metrics/precision(B)'] = last_row[col]
                elif 'precision' in col_lower and 'box' in col_lower:
                    metrics['metrics/precision(B)'] = last_row[col]
                # recall 찾기
                elif 'recall' in col_lower and '(b)' in col_lower:
                    metrics['metrics/recall(B)'] = last_row[col]
                elif 'recall' in col_lower and 'box' in col_lower:
                    metrics['metrics/recall(B)'] = last_row[col]
                # mAP50 찾기
                elif 'map50' in col_lower and 'map50-95' not in col_lower and '(b)' in col_lower:
                    metrics['metrics/mAP50(B)'] = last_row[col]
                elif 'map50' in col_lower and 'map50-95' not in col_lower and 'box' in col_lower:
                    metrics['metrics/mAP50(B)'] = last_row[col]
                # mAP50-95 찾기
                elif 'map50-95' in col_lower and '(b)' in col_lower:
                    metrics['metrics/mAP50-95(B)'] = last_row[col]
                elif 'map50-95' in col_lower and 'box' in col_lower:
                    metrics['metrics/mAP50-95(B)'] = last_row[col]
                elif 'map(' in col_lower and 'box' in col_lower and '50-95' in col_lower:
                    metrics['metrics/mAP50-95(B)'] = last_row[col]
            
            print(f"결과 CSV 파일에서 메트릭 읽기 완료: {results_csv_path}")
            print(f"  읽은 메트릭: {metrics}")
    else:
        # results 객체에서 직접 메트릭 추출 시도
        print("results.csv 파일을 찾을 수 없어 results 객체에서 메트릭 추출 시도...")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"  results.results_dict에서 추출: {metrics}")
        elif hasattr(results, 'metrics'):
            metrics = results.metrics
            print(f"  results.metrics에서 추출: {metrics}")
        elif hasattr(results, 'box'):
            # YOLO results 객체의 box 속성에서 메트릭 추출
            box_metrics = results.box
            metrics['metrics/precision(B)'] = getattr(box_metrics, 'p', None)
            metrics['metrics/recall(B)'] = getattr(box_metrics, 'r', None)
            metrics['metrics/mAP50(B)'] = getattr(box_metrics, 'map50', None)
            metrics['metrics/mAP50-95(B)'] = getattr(box_metrics, 'map', None)
            print(f"  results.box에서 추출: {metrics}")
        else:
            # results 객체의 속성 확인
            print(f"  results 객체 타입: {type(results)}")
            print(f"  results 객체 속성: {[attr for attr in dir(results) if not attr.startswith('_')]}")
            
            # dict-like 객체인 경우
            if isinstance(results, dict):
                metrics = results
                print(f"  results가 dict 타입: {metrics}")
            else:
                print("경고: results.csv 파일을 찾을 수 없고, results 객체에서도 메트릭을 추출할 수 없습니다.")
    
    return results, metrics

def save_results_to_csv(results, metrics, experiment_name, output_csv="test_results_summary.csv", verbose=True):
    """평가 결과를 CSV 파일로 저장"""
    if verbose:
        print("\n결과 저장 중...")
    
    # CSV 파일 생성
    output_path = Path(output_csv)
    file_exists = output_path.exists()
    
    with open(output_path, 'a', newline='') as f:
        fieldnames = [
            'experiment_name',
            'metrics/precision(B)',
            'metrics/recall(B)',
            'metrics/mAP50(B)',
            'metrics/mAP50-95(B)'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 헤더 작성 (파일이 새로 생성된 경우만)
        if not file_exists:
            writer.writeheader()
        
        # 메트릭 값 추출 (문자열을 float로 변환)
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                if isinstance(value, str):
                    if value == 'N/A' or value == '' or value.lower() == 'nan':
                        return default
                    return float(value)
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # 메트릭 키 찾기 (다양한 형식 지원)
        def find_metric_key(metrics_dict, possible_keys):
            for key in possible_keys:
                if key in metrics_dict:
                    return metrics_dict[key]
            return None
        
        # 결과 행 작성 (inference 결과만 저장)
        precision = find_metric_key(metrics, ['metrics/precision(B)', 'precision(B)', 'precision'])
        recall = find_metric_key(metrics, ['metrics/recall(B)', 'recall(B)', 'recall'])
        map50 = find_metric_key(metrics, ['metrics/mAP50(B)', 'mAP50(B)', 'map50'])
        map50_95 = find_metric_key(metrics, ['metrics/mAP50-95(B)', 'mAP50-95(B)', 'map50-95', 'map'])
        
        row = {
            'experiment_name': experiment_name,
            'metrics/precision(B)': round(safe_float(precision, 0.0), 5),
            'metrics/recall(B)': round(safe_float(recall, 0.0), 5),
            'metrics/mAP50(B)': round(safe_float(map50, 0.0), 5),
            'metrics/mAP50-95(B)': round(safe_float(map50_95, 0.0), 5)
        }
        
        writer.writerow(row)
    
    if verbose:
        print(f"결과가 {output_csv}에 저장되었습니다.")
        print("Test 데이터셋 평가 결과:")
        print(f"  Precision(B): {row['metrics/precision(B)']:.5f}")
        print(f"  Recall(B): {row['metrics/recall(B)']:.5f}")
        print(f"  mAP50(B): {row['metrics/mAP50(B)']:.5f}")
        print(f"  mAP50-95(B): {row['metrics/mAP50-95(B)']:.5f}")
    else:
        # 간단한 요약만 출력
        print(f"  저장 완료 - Precision: {row['metrics/precision(B)']:.5f}, "
              f"Recall: {row['metrics/recall(B)']:.5f}, "
              f"mAP50: {row['metrics/mAP50(B)']:.5f}, "
              f"mAP50-95: {row['metrics/mAP50-95(B)']:.5f}")

def main():
    """메인 함수"""
    # 현재 디렉토리를 스크립트 디렉토리로 설정
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    os.environ['PYTHONPATH'] = str(script_dir) + ':' + os.environ.get('PYTHONPATH', '')
    
    print("="*50)
    print("Test 데이터셋 평가 스크립트 (모든 run)")
    print("="*50)
    
    # 1. 모든 run 정보 로드
    df = load_all_runs()
    if df is None:
        return

    # 2. 결과 CSV 파일 초기화 (기존 파일이 있으면 백업)
    output_csv = "test_results_summary.csv"
    output_path = Path(output_csv)
    if output_path.exists():
        backup_path = Path(f"{output_csv}.backup")
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"\n기존 결과 파일을 백업했습니다: {backup_path}")
        # 새 파일로 시작 (기존 내용 덮어쓰기)
        output_path.unlink()

    # 3. 모든 run에 대해 반복
    total_runs = len(df)
    successful_runs = 0
    failed_runs = 0
    
    print(f"\n총 {total_runs}개의 run에 대해 평가를 시작합니다...")
    print("="*50)
    
    for idx, row in df.iterrows():
        experiment_name = row['experiment_name']
        run_num = idx + 1
        
        print(f"\n[{run_num}/{total_runs}] 처리 중: {experiment_name}")
        print("-" * 50)
        
        # 가중치 파일 찾기
        weight_path = find_weight_file(experiment_name)
        if weight_path is None:
            print(f"  ⚠️  가중치 파일을 찾을 수 없어 건너뜁니다.")
            failed_runs += 1
            continue
        
        # Test 데이터셋에 대해 평가
        try:
            results, metrics = evaluate_on_test(weight_path, experiment_name)
            if results is None or not metrics:
                print(f"  ⚠️  평가 실패: 메트릭을 추출할 수 없습니다.")
                failed_runs += 1
                continue
            
            # 결과를 CSV로 저장 (verbose=False로 간단한 출력만)
            save_results_to_csv(results, metrics, experiment_name, output_csv, verbose=False)
            successful_runs += 1
            
        except Exception as e:
            print(f"  ✗ 오류 발생: {str(e)}")
            failed_runs += 1
            import traceback
            traceback.print_exc()
            continue

    # 4. 최종 요약
    print("\n" + "=" * 50)
    print("모든 평가 완료!")
    print("=" * 50)
    print(f"총 run 수: {total_runs}")
    print(f"성공: {successful_runs}")
    print(f"실패: {failed_runs}")
    print(f"결과 파일: {output_csv}")
    print("=" * 50)

if __name__ == "__main__":
    main()
