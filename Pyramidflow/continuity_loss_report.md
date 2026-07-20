# Continuity Loss 적용 보고서 — PyramidFlow (원본 vs 변경)

- 작성일: 2026-06-17
- 데이터셋: MVTec AD 15개 카테고리
- 실험 조건: `alpha = 0.25`, `beta = 0.5`, `mode = l1`

---

## 0. 요약 (Executive Summary)

- **원본** PyramidFlow는 **Fourier loss 단독**으로 학습한다. 본 작업은 여기에 **공간 연속성(continuity) loss**를 추가했다(**변경**).
- **모델 아키텍처는 원본 그대로**이며, 바뀐 것은 ① 학습 손실에 continuity 항 추가, ② 관련 하이퍼파라미터, ③ 손실 항 분리 로깅이다.
- continuity loss는 `continuity.py`의 **`diff_loss_2d`** (feature map의 1차·2차 공간 차분)를 사용하며, 모델 feature 피라미드에 적용 후 Fourier loss에 더한다.
- 결과(15개 평균, 임계값 무관 지표):
  - **Image-AUROC: 0.8667 → 0.8751 (+0.85pp)** — 개선
  - Pixel-AUROC: 0.9574 → 0.9584 (+0.10pp) — 사실상 동일
  - Pixel-PRO: 0.9508 → 0.9486 (−0.22pp) — 사실상 동일
- 개선은 **이미지 단위 약점 카테고리에 집중**(capsule·toothbrush·wood·screw), 픽셀 단위는 무손상. 단 일부 쉬운 카테고리는 소폭 하락, 단일 seed라 재현 검증 필요.

---

## 1. 배경 및 목적

PyramidFlow는 정상 이미지만으로 학습하는 정규화 흐름(normalizing flow) 기반 이상 탐지 모델이다. 학습은 "정상 샘플끼리 feature가 같아지도록" 만들고, 정상 평균을 템플릿으로 두어 테스트 시 **템플릿과의 차이**로 이상을 탐지한다.

가설: 이상(스크래치·구멍·얼룩)은 **국소적 공간 연속성을 깨는** 지점이다. 따라서 정상 feature에 "공간적으로 매끄럽다"는 사전(prior)을 약하게 부여하면, 연속성을 깨는 이상 부위가 더 도드라져 **특히 이미지 단위 판정이 약한 카테고리**가 개선될 것으로 기대했다.

---

## 2. 원본과 달라진 점

| 구분 | 원본 | 변경 |
|---|---|---|
| 모델 구조 | PyramidFlow | **동일 (변경 없음)** |
| 학습 손실 | Fourier loss 단독 | Fourier + continuity(옵션) |
| 하이퍼파라미터 | — | `continuityAlpha/Beta/Mode` 추가 |
| 손실 모듈 | — | `continuity.py` 추가, `from continuity import diff_loss_2d` |

> 핵심: **추론(test.py)·모델(model.py)·데이터 파이프라인은 그대로**이고, 차이는 전적으로 "학습 손실에 continuity 항을 더할 수 있게 한 것"과 그 부수 설정이다. continuity를 끄면 분기를 타지 않아 원본과 동일하게 동작한다.

---

## 3. continuity.py에서 사용한 Loss

### 3.1 실제 사용 함수: `diff_loss_2d`

`train.py`의 래퍼 `pyramid_continuity_loss`가 각 피라미드 레벨에 `continuity.py`의 **`diff_loss_2d`**(`continuity.py:144`)를 적용하고 레벨 평균한다.

```python
# train.py:383-385
def pyramid_continuity_loss(pyramid, alpha=0.25, beta=0.5, mode="l1"):
    losses = [diff_loss_2d(feat, alpha=alpha, beta=beta, mode=mode) for feat in pyramid]
    return sum(losses) / len(losses)
```

### 3.2 `diff_loss_2d`의 정의

feature map `x = [B, C, H, W]`에 대해:

- **1차 차분(기울기, smoothness)**: 인접 픽셀 차이
  - 가로: `x[:,:,:,1:] - x[:,:,:,:-1]`, 세로: `x[:,:,1:,:] - x[:,:,:-1,:]`
- **2차 차분(곡률, curvature)**: 변화율의 변화
  - 가로: `x[:,:,:,2:] - 2·x[:,:,:,1:-1] + x[:,:,:,:-2]`, 세로 동일
- 축약(`mode`): `l1` = 절댓값 평균(edge-preserving), `l2` = 제곱평균제곱근
- 최종: **`alpha · (1차 차분) + beta · (2차 차분)`**

의미: "feature map을 공간적으로 매끄럽고(1차) 급격한 꺾임 없게(2차)" 만드는 정규화.

---

## 4. 입력된 인자

### 4.1 호출

```python
cont_loss = pyramid_continuity_loss(
    pyramid2,                    # 입력 텐서
    alpha=continuity_alpha,      # 0.25
    beta=continuity_beta,        # 0.5
    mode=continuity_mode,        # "l1"
)
```

| 인자 | 값 | 출처 | 의미 |
|---|---|---|---|
| `pyramid2` | 모델 feature 피라미드(4레벨) | `flow(image)` | **diff가 아닌 feature 그 자체** |
| `alpha` | 0.25 | `hyp.continuityAlpha` | 1차(매끄러움) 계수 |
| `beta` | 0.5 | `hyp.continuityBeta` | 2차(곡률) 계수 |
| `mode` | `l1` | `hyp.continuityMode` | 절댓값(edge-preserving) |

> 주의: `pyramid_continuity_loss`의 입력은 `diffes`(배치 쌍 차이)가 아니라 **`pyramid2`(feature)** 이다. Fourier loss(샘플 간 차이)와 continuity loss(feature 내부 매끄러움)는 서로 다른 대상을 본다.

---

## 5. 어디에 Loss를 더했는가

학습 루프 내 배치마다, `loss.backward()` 직전에 결합한다.

```python
base_loss = torch.fft.fft2(diff_pixel).abs().mean()   # ① 원본 Fourier loss
if loss_mode == 1:
    cont_loss = pyramid_continuity_loss(pyramid2, ...)  # ② continuity
    loss = base_loss + cont_loss                        # ③ 합산 ← 여기
else:
    cont_loss = base_loss.new_tensor(0.0)
    loss = base_loss                                    # 원본과 동일
```

- 결합식: **`loss = base_loss + cont_loss`** (alpha=0.25·beta=0.5가 continuity 항의 계수)
- 이후 `loss.backward()` → `clip_grad_norm_(max_norm=1.0)` → `optimizer.step()`로 두 항이 함께 역전파된다.

---

## 6. 실험 결과: 원본 vs 변경

> 비교는 MVTec 15개 카테고리, 동일 테스트셋(카테고리별 이미지 수 일치 확인). Pixel-AUROC / Image-AUROC / Pixel-PRO는 **임계값과 무관(threshold-free)** 한 지표이므로 모델 분리력 비교에 사용.

### 6.1 평균 (15개 카테고리)

| 지표 | 원본 | 변경 | Δ |
|---|---|---|---|
| Pixel-AUROC | 0.9574 | 0.9584 | **+0.10pp** |
| Pixel-PRO | 0.9508 | 0.9486 | **−0.22pp** |
| **Image-AUROC** | 0.8667 | 0.8751 | **+0.85pp** |

→ 픽셀 단위(localization)는 사실상 동일(over-smooth 손상 없음), **이미지 단위 판정만 평균 +0.85pp 개선**.

### 6.2 카테고리별 Image-AUROC (Δ 내림차순)

| category | 원본 | 변경 | Δ |
|---|---|---|---|
| toothbrush | 0.822 | 0.883 | **+6.11pp** |
| capsule | 0.794 | 0.855 | **+6.06pp** |
| wood | 0.929 | 0.969 | **+4.04pp** |
| screw | 0.582 | 0.618 | **+3.52pp** |
| transistor | 0.821 | 0.837 | +1.59pp |
| pill | 0.759 | 0.769 | +0.99pp |
| grid | 0.949 | 0.955 | +0.59pp |
| hazelnut | 0.913 | 0.915 | +0.25pp |
| leather | 0.978 | 0.977 | −0.07pp |
| zipper | 0.963 | 0.957 | −0.60pp |
| tile | 0.992 | 0.983 | −0.87pp |
| cable | 0.823 | 0.810 | −1.25pp |
| carpet | 0.898 | 0.876 | −2.13pp |
| metal_nut | 0.847 | 0.820 | −2.74pp |
| bottle | 0.930 | 0.902 | −2.78pp |

### 6.3 약점 카테고리의 미검출(FN) 감소 — 학습 임계값 기준

임계값 조정으로 해결되지 않던(분리력 자체가 낮던) 카테고리에서 놓치는 불량이 감소:

| category | 원본 FN → recall | 변경 FN → recall | 미검출 감소 |
|---|---|---|---|
| capsule | 53 → 0.51 | **33 → 0.70** | −20개 |
| pill | 47 → 0.67 | **32 → 0.77** | −15개 |
| screw | 85 → 0.29 | **72 → 0.40** | −13개 |

### 6.4 동일 FPR(운영점 보정) 종합

임계값 위치 차이를 제거하기 위해, 정상 점수 percentile로 FPR을 맞춘 뒤 비교:

| 운영점 | 원본 recall / FN합 | 변경 recall / FN합 |
|---|---|---|
| trained | 0.832 / 259 | 0.828 / **246** |
| test_good p95 | 0.615 / 511 | **0.639 / 485** |
| test_good p99 | 0.521 / 627 | **0.566 / 579** |

→ **동일 오탐률에서 변경이 일관되게 불량을 더 잡음(FN 합 감소).** raw 임계값 비교보다 robust한 신호.

### 6.5 카테고리별 미검(FN)·과검(FP) (원본 vs 변경)

이미지 단위, 학습 임계값(`raw_trained`) 기준. **미검(FN)** = 불량을 놓침, **과검(FP)** = 정상을 불량으로 오탐. 변경 열 괄호는 **변화량(변경−원본)**, 음수(−) = 감소.

| category | 원본 미검(FN) | 원본 과검(FP) | 변경 미검(FN) | 변경 과검(FP) |
|---|---|---|---|---|
| bottle | 1 | 12 | 0 (−1) | 18 (+6) |
| cable | 27 | 14 | 15 (−12) | 23 (+9) |
| capsule | 53 | 5 | 33 (−20) | 3 (−2) |
| carpet | 0 | 28 | 0 (0) | 28 (0) |
| grid | 1 | 14 | 1 (0) | 15 (+1) |
| hazelnut | 14 | 5 | 12 (−2) | 8 (+3) |
| leather | 0 | 32 | 0 (0) | 32 (0) |
| metal_nut | 1 | 20 | 44 (+43) | 2 (−18) |
| pill | 47 | 7 | 32 (−15) | 12 (+5) |
| screw | 85 | 8 | 72 (−13) | 10 (+2) |
| tile | 2 | 2 | 13 (+11) | 1 (−1) |
| toothbrush | 0 | 12 | 0 (0) | 12 (0) |
| transistor | 5 | 24 | 14 (+9) | 8 (−16) |
| wood | 13 | 3 | 3 (−10) | 3 (0) |
| zipper | 10 | 6 | 7 (−3) | 7 (+1) |

- 약점 카테고리 FN(미검) 감소: **capsule 53→33, pill 47→32, screw 85→72, wood 13→3**.
- ⚠️ 이 표의 FN/FP·변화량은 **참고용**이다. 원본/변경의 임계값(`raw_trained`) 위치가 달라 모델 효과와 임계값 효과가 섞여 있으므로(예: metal_nut FN +43), 순수 모델 개선은 **6.2(AUROC)·6.4(동일 FPR)** 가 더 정확하다.
- TN/TP 포함 전체 confusion 및 픽셀 단위는 `test_summary.csv` / `confusion_matrix.csv` 참조.

---

## 7. 결론 및 한계

### 7.1 결론

1. continuity loss(`diff_loss_2d`, alpha=0.25·beta=0.5)는 **이미지 단위 약점 카테고리(capsule·toothbrush·wood·screw)에서 분명한 개선**을 보였고, 미검출(FN)이 실질적으로 감소했다.
2. 픽셀 단위 지표(AUROC/PRO)는 거의 변화 없음 → **localization 손상 없이** 이미지 판정만 개선.
3. 가설("연속성을 깨는 이상은 매끄러움 prior로 더 도드라진다")과 방향이 일치.

### 7.2 한계

- **단일 seed** 실험 → 일부 카테고리의 ±1~2pp 변화는 노이즈일 수 있음. (단, capsule·toothbrush의 +6pp는 노이즈로 보기엔 큼)
- 일부 쉬운 카테고리(bottle −2.8, metal_nut −2.7, carpet −2.1)는 image-AUROC 하락.
- 평균 개선폭(+0.85pp)은 modest.

### 7.3 향후 과제

- **멀티 seed(3개+)** 재현 검증.
- alpha·beta 세기를 더 키우거나(예: 0.5/1.0) 줄여(예: 0.1/0.2) 성능 곡선 확보.
- 약점 카테고리에만 continuity를 강하게 주는 **카테고리별 차등 계수(alpha/beta)** 검토.
- 임계값 보정(과탐 TN=0 이슈)은 continuity와 별개 이슈이며, `image_threshold_percentile` 상향 또는 val-good 기준 재보정으로 분리 대응.
