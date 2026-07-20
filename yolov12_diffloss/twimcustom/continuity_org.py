# models/continuity.py
import torch
import torch.nn as nn
import math

import torch.nn.functional as F

# -------------------------------------
# 2d 연속성...
# -------------------------------------

class MaskedConv2d(nn.Conv2d):
    #2D causal conv: 현재 위치보다 위쪽/왼쪽 픽셀만 참조
    #PixelCNN 방식의 mask 적용
    def __init__(self, in_ch, out_ch, kernel_size, mask_type='B', **kwargs):
        super().__init__(in_ch, out_ch, kernel_size, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, h, w = self.weight.size()
        yc, xc = h // 2, w // 2

        # mask 설정: 아래쪽 제거, 현재 행에서는 오른쪽 제거
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc + (mask_type == 'A'):] = 0

    def forward(self, x):
        # 가중치에 mask 적용 (인과적 conv)
        # 가중치가 죽는버그...20251118
        #self.weight.data *= self.mask
        #return super().forward(x)
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)

class ContinuityHead2D(nn.Module):
    """
    2D causal conv 기반 연속성 예측기
    입력: [B, C, H, W]
    출력: z, z_hat, (H, W)
    """
    def __init__(self, in_ch, d=256, k=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d, 1)  # 1차원 flatt하지 않을것...
        self.pred = nn.Sequential(
            MaskedConv2d(d, d, k, padding=k//2, mask_type='B'),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 1)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        z: [B, D, H, W] — encoder feature (projected)
        z_hat: [B, D, H, W] — causal conv2d 예측 결과
        """
        z = self.proj(x)          # [B, D, H, W]
        z_hat = self.pred(z)      # [B, D, H, W]
        return z, z_hat, (z.shape[2], z.shape[3])

def npp_loss_2d(z, z_hat, norm_mask=None, mode='l1'):
    # z, z_hat: [B, D, H, W]
    # norm_mask: [B, 1, H, W]
    # causal conv에서는 첫 위치 (0,0)는 이전 정보가 없어 z_hat[0,0] = 0이 됨
    # 따라서 z[0,0]도 0으로 설정하여 일관성 유지
    z_masked = z.clone()
    z_masked[:, :, 0, 0] = 0.0  # 첫 위치 (0,0)를 0으로 설정
    
    if mode == 'l1':
        diff = (z_masked - z_hat).abs().mean(1, keepdim=True)  # [B,1,H,W]
    elif mode == 'cos':
        #a = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        #b = z_hat / (z_hat.norm(dim=1, keepdim=True) + 1e-6)
        #diff = (1.0 - (a * b).sum(1, keepdim=True))  # [B,1,H,W]
        a = z_masked / (z_masked.norm(dim=1, keepdim=True).clamp(min=1e-6))
        b = z_hat / (z_hat.norm(dim=1, keepdim=True).clamp(min=1e-6))
        diff = (1.0 - (a * b).sum(1, keepdim=True).clamp(-1.0,1.0))  # [B,1,H,W]
    else:
        raise ValueError("mode must be 'l1' or 'cos'")

    # 마스크 적용
    if norm_mask is not None:
        if norm_mask.shape[1] == 1:
            w = norm_mask
        else:
            w = norm_mask.unsqueeze(1)  # [B,1,H,W] 보장
        loss = (diff * w).sum() / (w.sum() + 1e-6)
    else:
        loss = diff.mean()
    return loss
    # [B, D, H, W] → [B, T, D]
    #B, D, H, W = z.shape
    #z = z.permute(0, 2, 3, 1).reshape(B, H*W, D)
    #z_hat = z_hat.permute(0, 2, 3, 1).reshape(B, H*W, D)

    #return npp_loss(z, z_hat, norm_mask, mode)

############################################################


class SeqProjector(nn.Module):
    # CxHxW -> DxT (D는 임베딩 축)
    def __init__(self, in_ch, d=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d, 1)

    def forward(self, x): # x: [B,C,H,W]
        x = self.proj(x)         # [B,D,H,W]
        B, D, H, W = x.shape
        x = x.view(B, D, H*W)    # [B,D,T]
        x = x.transpose(1, 2)    # [B,T,D]
        return x, (H, W)

class CausalConv1D(nn.Module):
    # 가벼운 causal conv 예측기
    def __init__(self, d=256, k=3, hidden=256):
        super().__init__()
        pad = k-1
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, k, padding=pad),  # causal-like: 나중에 crop
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, d, 1)
        )
        self.k = k

    def forward(self, z):  # z: [B,T,D]
        zt = z.transpose(1, 2)  # [B,D,T]
        y = self.net(zt)        # [B,D,T+pad]
        y = y[:, :, :zt.size(2)]  # crop to causal
        y = y.transpose(1, 2)   # [B,T,D]
        # 예측은 z_t를 z_{<t}로: 첫 토큰은 예측 불가 → 쉬프트 비교에서 제외
        return y

class ContinuityHead(nn.Module):
    def __init__(self, in_ch, d=256, k=3):
        super().__init__()
        self.proj = SeqProjector(in_ch, d)
        self.pred = CausalConv1D(d=d, k=k)

    def forward(self, x):  # x: [B,C,H,W]
        z, hw = self.proj(x)         # [B,T,D], (H,W)
        z_hat = self.pred(z)         # [B,T,D]
        return z, z_hat, hw

def npp_loss(z, z_hat, norm_mask=None, mode='l1'):
    # z, z_hat: [B,T,D]
    # norm_mask: [B,T] (정상=1, 결함/경계=0~0.5). None이면 전부 1
    if mode == 'l1':
        diff = (z[:,1:,:] - z_hat[:,1:,:]).abs().mean(-1)  # [B,T-1]
    elif mode == 'cos':
        a = z[:,1:,:] / (z[:,1:,:].norm(dim=-1, keepdim=True)+1e-6)
        b = z_hat[:,1:,:] / (z_hat[:,1:,:].norm(dim=-1, keepdim=True)+1e-6)
        diff = (1.0 - (a*b).sum(-1))                        # [B,T-1]
    else:
        raise ValueError('mode must be l1 or cos')

    if norm_mask is not None:
        w = norm_mask[:,1:]  # [B,T-1]
        loss = (diff * w).sum() / (w.sum()+1e-6)
    else:
        loss = diff.mean()

    return loss

def delta_map(z, z_hat, hw):
    # Δ = |z - z_hat| 평균 → HxW로 복원
    B, T, D = z.size()
    H, W = hw
    d = (z[:,1:,:] - z_hat[:,1:,:]).abs().mean(-1)  # [B,T-1]
    pad = torch.zeros(B,1, device=z.device, dtype=d.dtype)
    d = torch.cat([pad, d], dim=1)                   # 첫 토큰 0 padding
    d = d.view(B, H, W)                              # [B,H,W]
    return d
