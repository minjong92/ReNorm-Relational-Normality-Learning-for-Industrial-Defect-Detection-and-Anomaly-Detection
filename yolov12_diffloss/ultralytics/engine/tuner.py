# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection, instance
segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Example:
    Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
    ```
"""

import random
import shutil
import subprocess
import time

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks, colorstr, remove_colorstr, yaml_print, yaml_save
# from ultralytics.utils.plotting import plot_tune_results  # CSV 플롯팅 비활성화


class Tuner:
    """
    Class responsible for hyperparameter tuning of YOLO models.

    The class evolves YOLO model hyperparameters over a given number of iterations
    by mutating them according to the search space and retraining the model to evaluate their performance.

    Attributes:
        space (dict): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_csv (Path): Path to the CSV file where evolution logs are saved.

    Methods:
        _mutate(hyp: dict) -> dict:
            Mutates the given hyperparameters within the bounds specified in `self.space`.

        __call__():
            Executes the hyperparameter evolution across multiple iterations.

    Example:
        Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
        ```

        Tune with custom search space.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.tune(space={key1: val1, key2: val2})  # custom search space dictionary
        ```
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks=None):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict, optional): Configuration for hyperparameter evolution.
        """
        self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "scale": (0.0, 0.95),  # image scale (+/- gain)
            "shear": (0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0.0, 1.0),  # image flip left-right (probability)
            "bgr": (0.0, 1.0),  # image channel bgr (probability)
            "mosaic": (0.0, 1.0),  # image mixup (probability)
            "mixup": (0.0, 1.0),  # image mixup (probability)
            "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
            # NPP Loss 하이퍼파라미터 추가
            "npp_lambda_2d": (0.0, 0.10),  # 2D NPP Loss 가중치
            "npp_lambda_1d": (0.0, 0.10),  # 1D NPP Loss 가중치
            "npp_bbox_mask_weight": (0.1, 0.3),  # Bbox 내부 마스크 가중치
        }
        # NPP FPN sources는 선택지로 처리 (튜닝 시 별도 처리 필요)
        self.npp_fpn_sources_choices = args.pop("npp_fpn_sources_choices", ["15,18,21"])
        self.use_custom_train = args.pop("use_custom_train", False)  # train.py 사용 여부
        self.custom_train_script = args.pop("custom_train_script", "train.py")  # 커스텀 학습 스크립트 경로
        self.args = get_cfg(overrides=args)
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        self.args.name = None  # reset to not affect training directory
        self.tune_csv = self.tune_dir / "tune_results.csv"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)
        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _mutate(self, parent="single", n=5, mutation=0.8, sigma=0.2):
        """
        Mutates the hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
        if self.tune_csv.exists():  # if CSV file exists: select best hyps and mutate
            # Select parent(s)
            try:
                # 헤더 먼저 확인
                with open(self.tune_csv, 'r') as f:
                    headers = f.readline().strip().split(',')
                
                # CSV 파일을 수동으로 파싱 (컬럼 수 불일치 문제 해결)
                lines = []
                with open(self.tune_csv, 'r') as f:
                    f.readline()  # 헤더 스킵
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) > 0:
                            lines.append(parts)
                
                if not lines:
                    raise ValueError("No data rows in CSV")
                
                # 헤더 구조 파악
                has_npp_fpn = 'npp_fpn_sources' in headers
                metric_keys = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]
                metric_cols = [h for h in headers if h in metric_keys]
                num_metric_cols = len(metric_cols)
                
                # 하이퍼파라미터 시작 위치 계산
                hyp_start_idx = 1 + num_metric_cols
                hyp_end_idx = hyp_start_idx + len(self.space.keys())
                
                # 데이터 파싱 (컬럼 수가 다른 행도 처리)
                fitness_list = []
                hyperparams_list = []
                for parts in lines:
                    try:
                        if len(parts) > 0:
                            fitness = float(parts[0])
                            # 하이퍼파라미터 추출 (컬럼 수가 부족하면 기본값 사용)
                            hyp_row = []
                            for i, k in enumerate(self.space.keys()):
                                col_idx = hyp_start_idx + i
                                if col_idx < len(parts):
                                    try:
                                        hyp_row.append(float(parts[col_idx]))
                                    except (ValueError, IndexError):
                                        # 기본값 사용
                                        v = self.space[k]
                                        if isinstance(v, tuple) and len(v) >= 2:
                                            hyp_row.append((v[0] + v[1]) / 2.0)
                                        else:
                                            hyp_row.append(0.0)
                                else:
                                    # 컬럼이 없으면 기본값
                                    v = self.space[k]
                                    if isinstance(v, tuple) and len(v) >= 2:
                                        hyp_row.append((v[0] + v[1]) / 2.0)
                                    else:
                                        hyp_row.append(0.0)
                            
                            fitness_list.append(fitness)
                            hyperparams_list.append(hyp_row)
                    except (ValueError, IndexError):
                        continue  # 잘못된 행은 스킵
                
                if not fitness_list:
                    raise ValueError("No valid data rows in CSV")
                
                x_hyperparams = np.array(hyperparams_list)
                fitness = np.array(fitness_list)
                
                n = min(n, len(fitness))  # number of previous results to consider
                x_sorted = x_hyperparams[np.argsort(-fitness)][:n]  # top n mutations
                fitness_sorted = fitness[np.argsort(-fitness)][:n]
                
                # weights 계산 (fitness가 모두 같으면 균등 가중치 사용)
                if fitness_sorted.max() == fitness_sorted.min():
                    w = np.ones(n)
                else:
                    w = fitness_sorted - fitness_sorted.min() + 1e-6  # weights (sum > 0)
                
                if parent == "single" or len(x_sorted) == 1:
                    x = x_sorted[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == "weighted":
                    x = (x_sorted * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                r = np.random  # method
                r.seed(int(time.time()))
                g = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])  # gains 0-1
                ng = len(self.space)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)
                hyp = {k: float(x[i] * v[i]) for i, k in enumerate(self.space.keys())}
            except (ValueError, IndexError, Exception) as e:
                LOGGER.warning(f"{self.prefix}Error reading CSV file in _mutate: {e}. Using initial values.")
                # 에러 발생 시 초기값 사용으로 폴백
                hyp = {}
                for k in self.space.keys():
                    if hasattr(self.args, k):
                        hyp[k] = getattr(self.args, k)
                    else:
                        v = self.space[k]
                        if isinstance(v, tuple) and len(v) >= 2:
                            hyp[k] = (v[0] + v[1]) / 2.0
                        else:
                            hyp[k] = 0.0
        else:
            # 초기 하이퍼파라미터 설정 (self.args에 없으면 space의 중간값 사용)
            hyp = {}
            for k in self.space.keys():
                if hasattr(self.args, k):
                    hyp[k] = getattr(self.args, k)
                else:
                    # space의 범위에서 중간값을 기본값으로 사용
                    v = self.space[k]
                    if isinstance(v, tuple) and len(v) >= 2:
                        hyp[k] = (v[0] + v[1]) / 2.0  # 범위의 중간값
                    else:
                        hyp[k] = 0.0  # 기본값

        # Constrain to limits
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # lower limit
            hyp[k] = min(hyp[k], v[1])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits
        
        # NPP FPN sources는 선택지 중에서 선택 (진화 알고리즘 적용)
        if hasattr(self, 'npp_fpn_sources_choices') and self.npp_fpn_sources_choices:
            if self.tune_csv.exists():
                # 이전 결과에서 가중치 기반 선택 (fitness가 높을수록 선택 확률 증가)
                try:
                    with open(self.tune_csv, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 1:  # 헤더 + 최소 1개 데이터
                        headers = lines[0].strip().split(',')
                        if 'npp_fpn_sources' in headers:
                            idx = headers.index('npp_fpn_sources')
                            # 각 FPN sources 선택지별로 fitness 수집
                            fpn_fitness = {}  # {fpn_value: [fitness1, fitness2, ...]}
                            for line in lines[1:]:
                                parts = line.strip().split(',')
                                if len(parts) > idx:
                                    try:
                                        fitness = float(parts[0])
                                        fpn_value = parts[idx].strip()
                                        if fpn_value in self.npp_fpn_sources_choices:
                                            if fpn_value not in fpn_fitness:
                                                fpn_fitness[fpn_value] = []
                                            fpn_fitness[fpn_value].append(fitness)
                                    except:
                                        continue
                            
                            # 각 선택지의 평균 fitness 계산
                            if fpn_fitness:
                                fpn_avg_fitness = {k: sum(v) / len(v) for k, v in fpn_fitness.items()}
                                # 시도하지 않은 선택지도 포함 (작은 가중치 부여)
                                all_choices = list(self.npp_fpn_sources_choices)
                                weights = []
                                for choice in all_choices:
                                    if choice in fpn_avg_fitness:
                                        # 시도한 선택지: 실제 평균 fitness 사용
                                        weights.append(fpn_avg_fitness[choice])
                                    else:
                                        # 시도하지 않은 선택지: 작은 가중치 부여 (탐색 유도)
                                        # 시도한 선택지들의 최소 fitness를 사용하거나, 작은 양수값 사용
                                        min_tried_fitness = min(fpn_avg_fitness.values()) if fpn_avg_fitness else 0.0
                                        weights.append(max(0.0, min_tried_fitness * 0.5))  # 최소값의 50%로 탐색 유도
                                
                                # 음수 fitness 방지 (최소값을 0으로 조정)
                                min_weight = min(weights)
                                if min_weight < 0:
                                    weights = [w - min_weight + 1e-6 for w in weights]
                                
                                # weights가 모두 0이거나 합이 0인 경우 균등 가중치 사용
                                weights_sum = sum(weights)
                                if weights_sum <= 0 or all(w <= 0 for w in weights):
                                    weights = [1.0] * len(all_choices)  # 균등 가중치
                                
                                hyp['npp_fpn_sources'] = random.choices(all_choices, weights=weights)[0]
                            else:
                                hyp['npp_fpn_sources'] = random.choice(self.npp_fpn_sources_choices)
                        else:
                            hyp['npp_fpn_sources'] = random.choice(self.npp_fpn_sources_choices)
                    else:
                        hyp['npp_fpn_sources'] = random.choice(self.npp_fpn_sources_choices)
                except Exception as e:
                    LOGGER.warning(f"Error reading npp_fpn_sources from CSV: {e}")
                    hyp['npp_fpn_sources'] = random.choice(self.npp_fpn_sources_choices)
            else:
                hyp['npp_fpn_sources'] = random.choice(self.npp_fpn_sources_choices)

        return hyp

    def __call__(self, model=None, iterations=10, cleanup=True):
        """
        Executes the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the number of iterations, performing the following steps in each iteration:
        1. Load the existing hyperparameters or initialize new ones.
        2. Mutate the hyperparameters using the `mutate` method.
        3. Train a YOLO model with the mutated hyperparameters.
        4. Log the fitness score and mutated hyperparameters to a CSV file.

        Args:
           model (Model): A pre-initialized YOLO model to be used for training.
           iterations (int): The number of generations to run the evolution for.
           cleanup (bool): Whether to delete iteration weights to reduce storage space used during tuning.

        Note:
           The method utilizes the `self.tune_csv` Path object to read and log hyperparameters and fitness scores.
           Ensure this path is set correctly in the Tuner instance.
        """
        t0 = time.time()
        best_save_dir, best_metrics = None, None
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
        for i in range(iterations):
            # Mutate hyperparameters
            mutated_hyp = self._mutate()
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

            metrics = {}
            train_args = {**vars(self.args), **mutated_hyp}
            
            # NPP 하이퍼파라미터 분리
            npp_params = {}
            if 'npp_lambda_2d' in mutated_hyp:
                npp_params['npp_lambda_2d'] = mutated_hyp.pop('npp_lambda_2d')
            if 'npp_lambda_1d' in mutated_hyp:
                npp_params['npp_lambda_1d'] = mutated_hyp.pop('npp_lambda_1d')
            if 'npp_bbox_mask_weight' in mutated_hyp:
                npp_params['npp_bbox_mask_weight'] = mutated_hyp.pop('npp_bbox_mask_weight')
            if 'npp_fpn_sources' in mutated_hyp:
                npp_params['npp_fpn_sources'] = mutated_hyp.pop('npp_fpn_sources')
            
            save_dir = get_save_dir(get_cfg(train_args))
            weights_dir = save_dir / "weights"
            try:
                if self.use_custom_train:
                    # 커스텀 train.py 스크립트 사용
                    import os
                    script_path = os.path.join(os.getcwd(), self.custom_train_script)
                    if not os.path.isabs(script_path):
                        script_path = os.path.abspath(script_path)
                    script_dir = os.path.dirname(script_path) or os.getcwd()
                    cmd = ["python", script_path]
                    # 표준 파라미터를 명령줄 인자로 추가 (batch, device, epochs, imgsz, workers, amp)
                    standard_params = ['batch', 'device', 'epochs', 'imgsz', 'workers', 'amp']
                    for k in standard_params:
                        if k in train_args:
                            v = train_args[k]
                            if isinstance(v, bool):
                                cmd.append(f"--{k}={str(v).lower()}")
                            else:
                                cmd.append(f"--{k}={v}")
                    # NPP 파라미터를 명령줄 인자로 추가
                    for k, v in npp_params.items():
                        cmd.append(f"--{k}={v}")
                    LOGGER.info(f"{self.prefix}Running custom train script: {' '.join(cmd)}")
                    env = os.environ.copy()
                    env['PYTHONPATH'] = script_dir + ':' + env.get('PYTHONPATH', '')
                    return_code = subprocess.run(cmd, check=True, cwd=script_dir, env=env).returncode
                else:
                    # 기본 yolo train 명령 사용
                    cmd = ["yolo", "train", *(f"{k}={v}" for k, v in train_args.items())]
                    return_code = subprocess.run(" ".join(cmd), check=True, shell=True).returncode
                
                ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
                if ckpt_file.exists():
                    ckpt = torch.load(ckpt_file, map_location='cpu')
                    metrics = ckpt.get("train_metrics", {})
                assert return_code == 0, "training failed"

            except Exception as e:
                LOGGER.warning(f"WARNING ❌️ training failure for hyperparameter tuning iteration {i + 1}\n{e}")
                metrics = {}

            # Save results and mutated_hyp to CSV
            fitness = metrics.get("fitness", 0.0)
            
            # 주요 메트릭 추출 (Detection 모델의 경우: mAP50, mAP50-95, precision, recall)
            main_metrics = {}
            metric_keys = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]
            for key in metric_keys:
                if key in metrics:
                    main_metrics[key] = metrics[key]
            
            # 모든 하이퍼파라미터를 포함 (NPP 파라미터 포함)
            all_hyp = {**mutated_hyp, **npp_params}
            
            # 헤더 구조 확인 (CSV 파일이 있으면 기존 헤더 읽기, 없으면 새로 생성)
            if self.tune_csv.exists():
                with open(self.tune_csv, 'r') as f:
                    first_line = f.readline().strip()
                headers = first_line.split(',')
                has_npp_fpn_column = 'npp_fpn_sources' in headers
                # 기존 헤더에서 메트릭 키 확인
                has_main_metrics = any(key in headers for key in metric_keys)
            else:
                has_npp_fpn_column = hasattr(self, 'npp_fpn_sources_choices')
                has_main_metrics = len(main_metrics) > 0  # 메트릭이 있으면 포함
                # 헤더 생성: fitness + 주요 메트릭 + 하이퍼파라미터 + npp_fpn_sources
                header_list = ["fitness"]
                if has_main_metrics:
                    header_list.extend(metric_keys)
                header_list.extend(list(self.space.keys()))
                if has_npp_fpn_column:
                    header_list.append("npp_fpn_sources")
                headers = ",".join(header_list) + "\n"
            
            # 헤더 구조에 맞춰 log_row 생성 (항상 동일한 컬럼 수 유지)
            log_row = [round(fitness, 5)]
            if has_main_metrics:
                # 주요 메트릭 추가 (없으면 빈 문자열)
                for key in metric_keys:
                    log_row.append(round(main_metrics.get(key, 0.0), 5))
            log_row.extend([all_hyp.get(k, 0) for k in self.space.keys()])
            if has_npp_fpn_column:
                log_row.append(npp_params.get('npp_fpn_sources', ''))
            
            # 헤더는 첫 번째 iteration에만 작성
            header_line = "" if self.tune_csv.exists() else headers
            
            with open(self.tune_csv, "a") as f:
                f.write(header_line + ",".join(map(str, log_row)) + "\n")

            # Get best results
            try:
                x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
                # 컬럼 수가 일치하는지 확인
                if len(x.shape) > 1 and x.shape[1] > 0:
                    fitness = x[:, 0]  # first column
                    best_idx = fitness.argmax()
                else:
                    LOGGER.warning(f"{self.prefix}CSV file has inconsistent column count. Using current metrics as best.")
                    fitness = np.array([metrics.get("fitness", 0.0)])
                    best_idx = 0
            except (ValueError, IndexError) as e:
                LOGGER.warning(f"{self.prefix}Error reading CSV file: {e}. Using current metrics as best.")
                fitness = np.array([metrics.get("fitness", 0.0)])
                best_idx = 0
            best_is_current = best_idx == i
            if best_is_current:
                best_save_dir = save_dir
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}
                for ckpt in weights_dir.glob("*.pt"):
                    shutil.copy2(ckpt, self.tune_dir / "weights")
            elif cleanup:
                shutil.rmtree(weights_dir, ignore_errors=True)  # remove iteration weights/ dir to reduce storage space

            # Plot tune results (비활성화 - CSV 컬럼 불일치 문제로 인해)
            # plot_tune_results(self.tune_csv)

            # Save and print tune results
            header = (
                f"{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
                f"{self.prefix}Best fitness metrics are {best_metrics}\n"
                f"{self.prefix}Best fitness model is {best_save_dir}\n"
                f"{self.prefix}Best fitness hyperparameters are printed below.\n"
            )
            LOGGER.info("\n" + header)
            data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.space.keys())}
            yaml_save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            yaml_print(self.tune_dir / "best_hyperparameters.yaml")
