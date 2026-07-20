# YOLOv12 NPP Loss 하이퍼파라미터 튜닝 가이드

## 개요

이 디렉토리에는 YOLOv12 모델에 NPP Loss를 적용하여 학습할 때 사용할 수 있는 여러 튜닝 스크립트가 포함되어 있습니다.

## 튜닝 가능한 하이퍼파라미터

1. **NPP_LAMBDA_2D**: 2D NPP Loss 가중치 (기본값: 0.02)
   - 추천 범위: 0.01 ~ 0.05
   - 값이 클수록 2D 연속성 loss에 더 많은 가중치 부여

2. **NPP_LAMBDA_1D**: 1D NPP Loss 가중치 (기본값: 0.06)
   - 추천 범위: 0.03 ~ 0.10
   - 일반적으로 Lambda 2D보다 높게 설정
   - 값이 클수록 1D 연속성 loss에 더 많은 가중치 부여

3. **BBOX_MASK_WEIGHT**: Bbox 내부 마스크 가중치 (기본값: 0.3)
   - 추천 범위: 0.1 ~ 0.5
   - 값이 낮을수록 bbox 영역의 연속성 loss에 덜 집중
   - 값이 높을수록 bbox 영역의 연속성을 더 강조

4. **FPN_SOURCES**: 사용할 FPN 소스 레이어
   - `"14"`: P3만 사용 (가장 작은 스케일)
   - `"14,17"`: P3, P4 사용 (중간 스케일)
   - `"14,17,20"`: P3, P4, P5 모두 사용 (모든 스케일, 기본값)

## 튜닝 스크립트 종류

### 1. `train_npp_grid_search.sh` - 전체 그리드 서치
**용도**: 모든 하이퍼파라미터 조합을 체계적으로 테스트

**파라미터 범위**:
- Lambda 2D: 0.01, 0.02, 0.03, 0.05 (4개)
- Lambda 1D: 0.03, 0.06, 0.10 (3개)
- Bbox Mask: 0.1, 0.2, 0.3, 0.4, 0.5 (5개)
- FPN Sources: 3가지 조합 (3개)
- **총 조합 수: 4 × 3 × 5 × 3 = 180개**

**사용법**:
```bash
bash train_npp_grid_search.sh
```

**주의**: 매우 많은 시간이 소요될 수 있습니다.

### 2. `train_npp_single_param_search.sh` - 단일 파라미터 튜닝
**용도**: 한 번에 하나의 파라미터만 변경하며 각각의 영향 확인

**파라미터 범위**:
- Lambda 2D: 0.005, 0.01, 0.02, 0.03, 0.05, 0.08 (6개)
- Lambda 1D: 0.01, 0.03, 0.06, 0.10, 0.15, 0.20 (6개)
- Bbox Mask: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 (7개)
- FPN Sources: 3가지 조합 (3개)
- **총 조합 수: 6 + 6 + 7 + 3 = 22개**

**사용법**:
```bash
bash train_npp_single_param_search.sh
```

**장점**: 빠르게 각 파라미터의 영향을 파악할 수 있음

### 3. `train_npp_focused_search.sh` - 포커스 그리드 서치 (추천)
**용도**: 실용적인 범위로 제한하여 핵심 조합만 테스트

**파라미터 범위**:
- Lambda 2D: 0.01, 0.02, 0.03 (3개)
- Lambda 1D: 0.03, 0.06, 0.10 (3개)
  - Lambda 1D < Lambda 2D인 조합은 자동으로 스킵
- Bbox Mask: 0.2, 0.3, 0.4 (3개)
- FPN Sources: 3가지 조합 (3개)
- **총 조합 수: 약 27개 (스킵 제외)**

**사용법**:
```bash
bash train_npp_focused_search.sh
```

**장점**: 합리적인 시간 내에 실용적인 결과를 얻을 수 있음

### 4. `train_with_npp.sh` - 단일 실행
**용도**: 특정 하이퍼파라미터 조합으로 한 번만 실행

**사용법**:
```bash
# 환경 변수로 설정
NPP_LAMBDA_2D=0.02 \
NPP_LAMBDA_1D=0.06 \
BBOX_MASK_WEIGHT=0.3 \
FPN_SOURCES="14,17,20" \
bash train_with_npp.sh
```

## 결과 확인

각 스크립트는 실행 후 다음 파일들을 생성합니다:
- `npp_*_search_results.txt`: 실행 로그 및 결과 요약
- `runs/train_npp_*/`: 각 조합별 학습 결과 디렉토리

## 추천 사용 순서

1. **첫 번째**: `train_npp_single_param_search.sh` 실행
   - 각 파라미터의 영향 파악
   - 빠른 시간 내에 기본적인 트렌드 확인

2. **두 번째**: `train_npp_focused_search.sh` 실행
   - 실용적인 범위에서 최적 조합 탐색
   - 합리적인 시간 내에 좋은 결과 얻기

3. **세 번째**: 필요시 `train_npp_grid_search.sh` 실행
   - 더 세밀한 탐색이 필요한 경우
   - 충분한 시간과 리소스가 있는 경우

## 주의사항

- 각 학습은 상당한 시간이 소요될 수 있습니다 (데이터셋 크기에 따라 다름)
- GPU 메모리 사용량을 모니터링하세요
- 디스크 공간이 충분한지 확인하세요 (각 실행마다 체크포인트 저장)
- 실행 전 `data/data.yaml` 파일이 올바르게 설정되어 있는지 확인하세요

## 예제

### 빠른 테스트 (단일 실행)
```bash
NPP_LAMBDA_2D=0.02 \
NPP_LAMBDA_1D=0.06 \
BBOX_MASK_WEIGHT=0.3 \
FPN_SOURCES="14" \
bash train_with_npp.sh
```

### 중간 규모 탐색
```bash
bash train_npp_focused_search.sh
```

### 전체 탐색
```bash
bash train_npp_grid_search.sh
```
