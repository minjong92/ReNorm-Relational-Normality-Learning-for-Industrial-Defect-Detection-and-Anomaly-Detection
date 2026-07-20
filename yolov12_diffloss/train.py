from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import sys

# 명령줄 인자에서 하이퍼파라미터 파싱
def parse_args():
    """명령줄 인자에서 차분로스(Diff Loss) 관련 하이퍼파라미터 및 표준 파라미터 파싱"""
    # 차분로스 파라미터 (기본값: 0으로 설정하여 비활성화)
    diff_alpha = 0.2  # 1차 차분 가중치
    diff_beta = 0.8   # 2차 차분 가중치
    npp_fpn_sources = "14,17,20"  # 기본값 (v12_diff에서는 14,17,20 사용)
    
    # NPP loss 파라미터 (차분로스와 호환성을 위해 유지, 기본값 0)
    npp_lambda_2d = 0.0
    npp_lambda_1d = 0.0
    npp_bbox_mask_weight = 0.3
    
    # 표준 파라미터 기본값
    batch = 1
    device = "0"
    epochs = 200
    imgsz = 1280
    workers = 0
    amp = False
    data = './data/data.yaml'
    project = 'runs/detect'
    
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
            amp = arg.lower() == 'true'
        elif arg.startswith('--data='):
            data = arg.split('=')[1]
        elif arg.startswith('--project='):
            project = arg.split('=')[1]
    
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
        'data': data,
        'project': project
    }

# 하이퍼파라미터 파싱
hyperparams = parse_args()

# 모델 로드 (YOLOv12)
# v12_diff 실험에서는 yolov12m.pt 가중치를 사용
model = YOLO('yolov12m.pt')  # .pt 파일만 지정하면 자동으로 yaml과 가중치 로드

# 차분로스 하이퍼파라미터를 별도로 저장
diff_params = {
    'diff_alpha': hyperparams['diff_alpha'],
    'diff_beta': hyperparams['diff_beta'],
    'npp_fpn_sources': hyperparams['npp_fpn_sources']
}

# NPP Loss 하이퍼파라미터도 저장 (호환성을 위해)
npp_params = {
    'npp_lambda_2d': hyperparams['npp_lambda_2d'],
    'npp_lambda_1d': hyperparams['npp_lambda_1d'],
    'npp_bbox_mask_weight': hyperparams['npp_bbox_mask_weight'],
    'npp_fpn_sources': hyperparams['npp_fpn_sources']
}

# 하이퍼파라미터 기반 프로젝트 이름 생성
FPN_STR = hyperparams['npp_fpn_sources'].replace(',', '_')
# 차분로스가 활성화된 경우 차분로스 이름 사용
if hyperparams['diff_alpha'] > 0.0 or hyperparams['diff_beta'] > 0.0:
    PROJECT_NAME = f"train_diff_alpha{hyperparams['diff_alpha']}_beta{hyperparams['diff_beta']}_fpn{FPN_STR}"
else:
    PROJECT_NAME = f"train_npp_l2d{hyperparams['npp_lambda_2d']}_l1d{hyperparams['npp_lambda_1d']}_mask{hyperparams['npp_bbox_mask_weight']}_fpn{FPN_STR}"

# overrides에는 표준 인자만 포함 (NPP 인자는 check_dict_alignment에서 에러 발생)
# 명령줄 인자에서 받은 값 사용 (없으면 기본값)
overrides = {
    'data': hyperparams.get('data', './data/data.yaml'),
    'epochs': hyperparams.get('epochs', 400),
    'batch': hyperparams.get('batch', 2),
    'imgsz': hyperparams.get('imgsz', 1280),
    'scale': 0.5,
    'mosaic': 0.3,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'device': hyperparams.get('device', "3"),
    'workers': hyperparams.get('workers', 0),
    'amp': hyperparams.get('amp', False),  # AMP 비활성화 (Segmentation fault 방지)
    'project': hyperparams.get('project', 'runs/detect'),  # 프로젝트 디렉토리
    'name': PROJECT_NAME,  # 하이퍼파라미터 기반 이름
    'exist_ok': True  # 같은 이름의 디렉토리가 있으면 덮어쓰기 (추가 '2' 생성 방지)
}

# 커스텀 trainer 클래스 정의
class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 차분로스 하이퍼파라미터를 args에 직접 추가 (check_dict_alignment 이후)
        for key, value in diff_params.items():
            setattr(self.args, key, value)
        # NPP Loss 하이퍼파라미터도 추가 (호환성을 위해)
        for key, value in npp_params.items():
            setattr(self.args, key, value)
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation with custom args filtered out."""
        from copy import copy
        from ultralytics.models import yolo
        
        # 커스텀 관련 인자를 제거한 args 복사본 생성
        validator_args = copy(self.args)
        custom_arg_names = ['diff_alpha', 'diff_beta', 'npp_lambda_2d', 'npp_lambda_1d', 'npp_bbox_mask_weight', 'npp_fpn_sources']
        for custom_arg in custom_arg_names:
            if hasattr(validator_args, custom_arg):
                delattr(validator_args, custom_arg)
        
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=validator_args, _callbacks=self.callbacks
        )

# YOLO 모델의 train() 메서드를 사용하되, 커스텀 trainer 전달
results = model.train(trainer=CustomDetectionTrainer, **overrides)

# Evaluate model performance on the validation set
# 학습 중 이미 검증이 수행되므로, 별도의 val() 호출은 선택사항입니다.
# 같은 디렉터리를 사용하도록 project와 name을 명시적으로 전달
metrics = model.val(
    project=overrides['project'],
    name=PROJECT_NAME,
    exist_ok=True
)

# Perform object detection on an image
#results = model("path/to/image.jpg")
#results[0].show()