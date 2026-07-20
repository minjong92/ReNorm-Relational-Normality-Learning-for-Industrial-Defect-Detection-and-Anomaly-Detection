import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A

from skimage import measure
from sklearn.metrics import auc

import time, os, logging, random
import os.path as osp
from glob import glob
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = ('bmp', 'png', 'jpg', 'jpeg', 'tif', 'tiff')


def collect_images(root, recursive=False):
    files = []
    if not root or not osp.isdir(root):
        return files
    pattern = '**/*' if recursive else '*'
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob(osp.join(root, f'{pattern}.{ext}'), recursive=recursive))
        files.extend(glob(osp.join(root, f'{pattern}.{ext.upper()}'), recursive=recursive))
    return sorted(set(files))


def first_existing_dir(paths):
    for path in paths:
        if osp.isdir(path):
            return path
    return None


def split_normal_files(files, val_ratio=0.1):
    if len(files) <= 1:
        return files, files
    val_count = max(1, int(round(len(files) * float(val_ratio))))
    val_count = min(val_count, len(files) - 1)
    return files[:-val_count], files[-val_count:]


def fix_randseed(seed):
    """ 
        Set random seeds for reproducibility 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)# use deterministic algorithms.
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)

    loader_dict = { 'worker_init_fn':seed_worker, 'generator':g}
    return loader_dict

class PadToSquare:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        img = TF.resize(img, (new_h, new_w))

        # Pad to center
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top

        img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        return img

class MOAIDataloader(Dataset):
    """
        MOAI Dataset.
        x_size is input image size, y_size is output mask size.
    """
    def __init__(self, mode='test', x_size=1024, y_size=256, datapath='../moai', hyp=None) -> None:
        super().__init__()
        self.mode = mode
        self.datapath = osp.abspath(datapath)
        self.test_root = None
        self.img_mean = [0.485, 0.456, 0.406] 
        self.img_std = [0.229, 0.224, 0.225]

        val_ratio = float(getattr(hyp, 'valRatio', 0.1)) if hyp is not None else 0.1
        val_dir = first_existing_dir([
            osp.join(self.datapath, 'validation', 'good'),
            osp.join(self.datapath, 'valid', 'good'),
            osp.join(self.datapath, 'val', 'good'),
        ])

        if mode == 'train':
            train_dir = first_existing_dir([
                osp.join(self.datapath, 'train', 'good'),
                osp.join(self.datapath, 'good'),
            ])
            train_files = collect_images(train_dir)
            if val_dir is None:
                self.files, _ = split_normal_files(train_files, val_ratio=val_ratio)
            else:
                self.files = train_files
            self.aug = A.Compose([
                        A.HorizontalFlip(p=hyp.hflip), A.VerticalFlip(p=hyp.vflip), A.RandomRotate90(p=hyp.rotate90),
                        A.ColorJitter(brightness=hyp.brightness, contrast=hyp.contrast, saturation=hyp.saturation, hue=hyp.hue)
                        #A.RGBShift(p=0.5),
                        #A.CLAHE(p=0.5),
                        #A.ChannelShuffle(p=0.5)
                        ])
        elif mode == 'val':
            if val_dir is not None:
                self.files = collect_images(val_dir)
            else:
                train_dir = first_existing_dir([
                    osp.join(self.datapath, 'train', 'good'),
                    osp.join(self.datapath, 'good'),
                ])
                _, self.files = split_normal_files(collect_images(train_dir), val_ratio=val_ratio)
        elif mode == 'test':
            self.test_root = first_existing_dir([osp.join(self.datapath, 'test')])
            if self.test_root is not None:
                self.files = collect_images(self.test_root, recursive=True)
            else:
                self.test_root = self.datapath
                self.files = collect_images(self.datapath, recursive=True)
        else:
            raise ValueError(f"Unknown dataset mode: {mode}")

        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found for mode='{mode}' under '{self.datapath}'")

        self.img2tensor = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((x_size,x_size)),
                                            # PadToSquare(x_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)
                                            ])
        self.label2tensor = transforms.Compose([transforms.ToPILImage(), 
                                                transforms.Resize((y_size,y_size)),
                                                transforms.ToTensor()])

    def __getitem__(self, index):
        # neg_files
        file_path = self.files[index]

        fname = self._sample_name(file_path)
        
        img = np.array(Image.open(file_path).convert('RGB'))
        label = np.zeros(img.shape[:2], dtype=np.float32)# h,w
        if self.mode == 'test':
            label_path = self._find_mask_path(file_path)
            if osp.exists(label_path):
                label = np.array(Image.open(label_path).convert('L'), dtype=np.float32)/255.
        if self.mode == 'train' :
            augresults = self.aug(image=img, mask=label)
            img, label = augresults['image'], augresults['mask']
        img, label = self.img2tensor(img), self.label2tensor(label)
        return {'images': img, 'labels': label.round().int(), 'fname': fname}

    def _sample_name(self, file_path):
        if self.mode == 'test' and self.test_root is not None:
            rel = osp.relpath(file_path, self.test_root)
            rel_no_ext = osp.splitext(rel)[0]
            return rel_no_ext.replace('\\', '_').replace('/', '_')
        return osp.splitext(osp.basename(file_path))[0]

    def _find_mask_path(self, file_path):
        if self.test_root is not None:
            rel = osp.relpath(file_path, self.test_root)
            parts = rel.replace('\\', '/').split('/')
            if parts and parts[0].lower() == 'good':
                return ''

            rel_dir = osp.dirname(rel)
            stem = osp.splitext(osp.basename(file_path))[0]
            gt_root = osp.join(self.datapath, 'ground_truth')
            candidates = [
                osp.join(gt_root, rel_dir, f'{stem}_mask.png'),
                osp.join(gt_root, rel_dir, f'{stem}.png'),
                osp.join(gt_root, rel_dir, stem, '000.png'),
            ]

            nested_masks = sorted(glob(osp.join(gt_root, rel_dir, stem, '*.png')))
            candidates.extend(nested_masks)

            for candidate in candidates:
                if osp.exists(candidate):
                    return candidate

        legacy_path = file_path.replace('valid', 'ground_truth')
        legacy_path = osp.splitext(legacy_path)[0] + '_mask.png'
        return legacy_path

    def __len__(self):
        return len(self.files)


def kornia_filter2d(input, kernel):
    """
        conv2d function from kornia.
    """
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    tmp_kernel = tmp_kernel.expand(-1, 64, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output


class BatchDiffLoss(nn.Module):
    """
        Difference Loss within a batch.
    """
    def __init__(self, batchsize, p=2) -> None:
        super().__init__()
        self.idx0, self.idx1 = np.triu_indices(n=batchsize, k=1)
        self.p = p
    def forward(self, pyramid):
        diffes = []
        for input in pyramid:
            diff = (input[self.idx0] - input[self.idx1]).abs()**self.p
            diffes.append(diff)
        return diffes

def getLogger(work_dir):
    time_now = time.strftime('%Y-%m-%d-%H.%M', time.localtime())
    logger = logging.getLogger(work_dir+f'_{time_now}')
    logger.setLevel(logging.INFO)  # Logging level
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    # FileHandler
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(osp.join(work_dir, f"{time_now}.txt"), mode='w')
    fHandler.setLevel(logging.DEBUG) 
    fHandler.setFormatter(formatter) 
    logger.addHandler(fHandler) 
    return logger, time_now


def compute_pro_score_fast(_amaps, _masks):
    """
        Compute PRO score fastly
            amaps: b,h,w
            masks: b,h,w
    """
    amaps = (_amaps-_amaps.min())/(_amaps.max()-_amaps.min())
    amaps, masks = amaps.cpu().numpy(), _masks.cpu().numpy()
    pro_lst, fpr_lst = [], []
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    max_step = 200
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / max_step
    weights, num_pros = [], 0
    for mask in masks:
        label = measure.label(mask, background=0)
        weight = np.zeros_like(label, dtype=np.float32)
        for k in range(1, np.max(label)+1):
            weight_mask = (label==k)
            weight[weight_mask] = 1/np.sum(weight_mask)
        weights.append(weight)
        num_pros += np.max(label)
    weights = np.stack(weights, axis=0)/num_pros
    inverse_masks = 1 - masks
    for th in np.arange(min_th, max_th, delta):
        binary_amaps = np.where(amaps>th, 1, 0)
        pro = (binary_amaps*weights).sum()
        pro_lst.append(pro)
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()
        fpr_lst.append(np.mean(fpr))
    return auc(fpr_lst, pro_lst)
