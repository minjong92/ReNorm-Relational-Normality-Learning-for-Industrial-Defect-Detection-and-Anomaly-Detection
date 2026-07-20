from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import sys
import itertools
import gc
import torch


def parse_args():
    """표준 파라미터만 파싱 (diff grid는 코드에서 loop로 처리)"""
    batch = 1
    device = "0"
    epochs = 100
    imgsz = 1280
    workers = 0
    amp = False
    data = "./data/data.yaml"
    project = "runs/detect"
    exist_ok = True

    # 부분 실행 옵션 (재시작/분할 실행용)
    start = 0
    end = None  # None이면 끝까지

    # fpn sources (diff 실험용)
    npp_fpn_sources = "14,17,20"

    for arg in sys.argv[1:]:
        if arg.startswith("--batch="):
            batch = int(arg.split("=", 1)[1])
        elif arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        elif arg.startswith("--epochs="):
            epochs = int(arg.split("=", 1)[1])
        elif arg.startswith("--imgsz="):
            imgsz = int(arg.split("=", 1)[1])
        elif arg.startswith("--workers="):
            workers = int(arg.split("=", 1)[1])
        elif arg.startswith("--amp="):
            amp = arg.split("=", 1)[1].lower() == "true"
        elif arg.startswith("--data="):
            data = arg.split("=", 1)[1]
        elif arg.startswith("--project="):
            project = arg.split("=", 1)[1]
        elif arg.startswith("--exist_ok="):
            exist_ok = arg.split("=", 1)[1].lower() == "true"
        elif arg.startswith("--start="):
            start = int(arg.split("=", 1)[1])
        elif arg.startswith("--end="):
            end = int(arg.split("=", 1)[1])
        elif arg.startswith("--diff_fpn_sources=") or arg.startswith("--npp_fpn_sources="):
            npp_fpn_sources = arg.split("=", 1)[1]

    return {
        "batch": batch,
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "workers": workers,
        "amp": amp,
        "data": data,
        "project": project,
        "exist_ok": exist_ok,
        "start": start,
        "end": end,
        "npp_fpn_sources": npp_fpn_sources,
    }


class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, diff_params=None, npp_params=None, **kwargs):
        self._diff_params = diff_params or {}
        self._npp_params = npp_params or {}
        super().__init__(*args, **kwargs)

        # check_dict_alignment 이후에 args에 직접 추가
        for k, v in self._diff_params.items():
            setattr(self.args, k, v)
        for k, v in self._npp_params.items():
            setattr(self.args, k, v)

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation with custom args filtered out."""
        from copy import copy
        from ultralytics.models import yolo

        validator_args = copy(self.args)
        custom_arg_names = [
            "diff_alpha", "diff_beta",
            "npp_lambda_2d", "npp_lambda_1d", "npp_bbox_mask_weight",
            "npp_fpn_sources"
        ]
        for a in custom_arg_names:
            if hasattr(validator_args, a):
                delattr(validator_args, a)

        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=validator_args, _callbacks=self.callbacks
        )


def main():
    args = parse_args()

    # ✅ 수정된 diff grid
    alpha_values = [0.0, 0.1, 0.3, 0.5]
    beta_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    grid = list(itertools.product(alpha_values, beta_values))  # (alpha, beta)

    total_combos = len(grid)

    start = max(0, args["start"])
    end = args["end"] if args["end"] is not None else total_combos
    end = min(end, total_combos)
    grid = grid[start:end]

    print(f"[INFO] Total combos: {total_combos} | Running slice: {start} ~ {end-1} (count={len(grid)})")

    # ✅ 모델 로드 (한 번만)
    model = YOLO("yolov12m.pt")

    # 공통 overrides (커스텀 인자 제외)
    base_overrides = {
        "data": args["data"],
        "epochs": args["epochs"],
        "batch": args["batch"],
        "imgsz": args["imgsz"],
        "scale": 0.9,
        "mosaic": 0.6,
        "mixup": 0.15,
        "copy_paste": 0.4,
        "device": args["device"],
        "workers": args["workers"],
        "amp": args["amp"],
        "project": args["project"],
        "exist_ok": args["exist_ok"],
    }

    fpn_sources = args["npp_fpn_sources"]
    fpn_str = fpn_sources.replace(",", "_")

    for idx, (diff_alpha, diff_beta) in enumerate(grid, start=start):
        # diff params
        diff_params = {
            "diff_alpha": float(diff_alpha),
            "diff_beta": float(diff_beta),
            "npp_fpn_sources": fpn_sources,  # 기존 구조 유지
        }

        # npp params (호환용, 기본 0 고정)
        npp_params = {
            "npp_lambda_2d": 0.0,
            "npp_lambda_1d": 0.0,
            "npp_bbox_mask_weight": 0.3,
            "npp_fpn_sources": fpn_sources,
        }

        name = f"train_diff_alpha{diff_alpha:.1f}_beta{diff_beta:.1f}_fpn{fpn_str}"
        overrides = dict(base_overrides)
        overrides["name"] = name

        print("\n" + "=" * 80)
        print(f"[RUN {idx+1:02d}/{total_combos}] name={name}")
        print(f"  - diff_alpha={diff_alpha:.1f}, diff_beta={diff_beta:.1f}, fpn_sources={fpn_sources}")
        print("=" * 80)

        try:
            # ✅ 학습
            model.train(
                trainer=lambda *t_args, **t_kwargs: CustomDetectionTrainer(
                    *t_args, diff_params=diff_params, npp_params=npp_params, **t_kwargs
                ),
                **overrides,
            )

            # ✅ 조합별 최종 val 저장
            model.val(
                project=overrides["project"],
                name=name,
                exist_ok=True,
            )

        except Exception as e:
            print(f"[ERROR] Failed combo name={name}: {e}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n[INFO] All experiments finished.")


if __name__ == "__main__":
    main()