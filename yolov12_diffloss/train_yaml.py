from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import sys
import os
from copy import copy


# -----------------------------
# 명령줄 인자 파싱
# -----------------------------
def parse_args():
    """차분로스(Diff Loss) / NPP Loss / 표준 파라미터 파싱"""
    diff_alpha = 0.0
    diff_beta = 0.0
    npp_fpn_sources = "14,17,20"

    npp_lambda_2d = 0.00
    npp_lambda_1d = 0.00
    npp_bbox_mask_weight = 0.0

    batch = 1
    device = "0"
    epochs = 200
    imgsz = 1280
    workers = 0
    amp = False
    project = "runs/detect"
    patience = 200   # 연구 실험이면 early stopping 끄는 걸 추천

    for arg in sys.argv[1:]:
        if arg.startswith('--npp_alpha=') or arg.startswith('--diff_alpha='):
            diff_alpha = float(arg.split('=')[1])
        elif arg.startswith('--npp_beta=') or arg.startswith('--diff_beta='):
            diff_beta = float(arg.split('=')[1])
        elif arg.startswith('--npp_fpn_sources=') or arg.startswith('--diff_fpn_sources='):
            npp_fpn_sources = arg.split('=')[1]
        elif arg.startswith('--npp_lambda_2d='):
            npp_lambda_2d = float(arg.split('=')[1])
        elif arg.startswith('--npp_lambda_1d='):
            npp_lambda_1d = float(arg.split('=')[1])
        elif arg.startswith('--npp_bbox_mask_weight='):
            npp_bbox_mask_weight = float(arg.split('=')[1])
        elif arg.startswith('--batch='):
            batch = int(arg.split('=')[1])
        elif arg.startswith('--device='):
            device = arg.split('=')[1]
        elif arg.startswith('--epochs='):
            epochs = int(arg.split('=')[1])
        elif arg.startswith('--imgsz='):
            imgsz = int(arg.split('=')[1])
        elif arg.startswith('--workers='):
            workers = int(arg.split('=')[1])
        elif arg.startswith('--amp='):
            amp = arg.split('=')[1].lower() == 'true'
        elif arg.startswith('--project='):
            project = arg.split('=')[1]
        elif arg.startswith('--patience='):
            patience = int(arg.split('=')[1])

    return {
        'diff_alpha': diff_alpha,
        'diff_beta': diff_beta,
        'npp_lambda_2d': npp_lambda_2d,
        'npp_lambda_1d': npp_lambda_1d,
        'npp_bbox_mask_weight': npp_bbox_mask_weight,
        'npp_fpn_sources': npp_fpn_sources,
        'batch': batch,
        'device': device,
        'epochs': epochs,
        'imgsz': imgsz,
        'workers': workers,
        'amp': amp,
        'project': project,
        'patience': patience
    }


hyperparams = parse_args()

# 차분로스 파라미터
diff_params = {
    'diff_alpha': hyperparams['diff_alpha'],
    'diff_beta': hyperparams['diff_beta'],
    'npp_fpn_sources': hyperparams['npp_fpn_sources']
}

# NPP Loss 파라미터
npp_params = {
    'npp_lambda_2d': hyperparams['npp_lambda_2d'],
    'npp_lambda_1d': hyperparams['npp_lambda_1d'],
    'npp_bbox_mask_weight': hyperparams['npp_bbox_mask_weight'],
    'npp_fpn_sources': hyperparams['npp_fpn_sources']
}


# -----------------------------
# 커스텀 trainer
# -----------------------------
class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in diff_params.items():
            setattr(self.args, key, value)

        for key, value in npp_params.items():
            setattr(self.args, key, value)

    def get_validator(self):
        from ultralytics.models import yolo

        validator_args = copy(self.args)
        custom_arg_names = [
            'diff_alpha', 'diff_beta',
            'npp_lambda_2d', 'npp_lambda_1d',
            'npp_bbox_mask_weight', 'npp_fpn_sources'
        ]
        for custom_arg in custom_arg_names:
            if hasattr(validator_args, custom_arg):
                delattr(validator_args, custom_arg)

        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks
        )


# -----------------------------
# 데이터셋 목록
# -----------------------------
dataset_yaml_list = [
    "./data/data0.1.yaml",
    "./data/data0.25.yaml",
    "./data/data0.5.yaml",
    "./data/data0.75.yaml",
]


# -----------------------------
# 프로젝트 이름 공통 prefix
# -----------------------------
FPN_STR = hyperparams['npp_fpn_sources'].replace(',', '_')

if hyperparams['diff_alpha'] > 0.0 or hyperparams['diff_beta'] > 0.0:
    base_project_name = f"train_diff_alpha{hyperparams['diff_alpha']}_beta{hyperparams['diff_beta']}_fpn{FPN_STR}"
else:
    base_project_name = f"train_npp_l2d{hyperparams['npp_lambda_2d']}_l1d{hyperparams['npp_lambda_1d']}_mask{hyperparams['npp_bbox_mask_weight']}_fpn{FPN_STR}"


# -----------------------------
# 순차 학습
# -----------------------------
for data_yaml in dataset_yaml_list:
    dataset_tag = os.path.splitext(os.path.basename(data_yaml))[0]   # dataset0.1
    run_name = f"{base_project_name}_{dataset_tag}"

    print(f"\n{'='*80}")
    print(f"[INFO] Start training: {data_yaml}")
    print(f"[INFO] Run name: {run_name}")
    print(f"{'='*80}\n")

    # 각 데이터셋 실험마다 동일한 pretrained weight에서 시작
    model = YOLO("yolov12m.pt")

    overrides = {
        'data': data_yaml,
        'epochs': hyperparams['epochs'],
        'patience': hyperparams['patience'],
        'batch': hyperparams['batch'],
        'imgsz': hyperparams['imgsz'],
        'scale': 0.9,
        'mosaic': 0.6,
        'mixup': 0.15,
        'copy_paste': 0.4,
        'device': hyperparams['device'],
        'workers': hyperparams['workers'],
        'amp': hyperparams['amp'],
        'project': hyperparams['project'],
        'name': run_name,
        'exist_ok': True,
    }

    results = model.train(trainer=CustomDetectionTrainer, **overrides)

    # 추가 검증이 꼭 필요할 때만 사용
    metrics = model.val(
        data=data_yaml,
        project=hyperparams['project'],
        name=run_name,
        exist_ok=True
    )

    print(f"[INFO] Finished: {data_yaml}")