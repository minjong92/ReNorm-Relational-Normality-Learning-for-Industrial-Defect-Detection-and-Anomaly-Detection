import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_tensor(x, name="tensor"):
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be torch.Tensor or None, got {type(x)}")
    return x


def _to_1ch_mask(norm_mask):
    norm_mask = _require_tensor(norm_mask, "norm_mask")
    if norm_mask is None:
        return None

    if norm_mask.dim() == 3:
        return norm_mask.unsqueeze(1)
    if norm_mask.dim() == 4 and norm_mask.size(1) == 1:
        return norm_mask
    if norm_mask.dim() == 4 and norm_mask.size(1) != 1:
        return norm_mask.mean(1, keepdim=True)

    raise ValueError(
        f"norm_mask must be [B,H,W] or [B,1,H,W] or [B,C,H,W], got {tuple(norm_mask.shape)}"
    )


def resize_mask_to_feature(mask_1ch, H, W):
    if mask_1ch is None:
        return None
    return F.interpolate(mask_1ch.float(), size=(H, W), mode="nearest")


def flatten_mask_hw(mask_1ch_hw):
    if mask_1ch_hw is None:
        return None
    B, _, H, W = mask_1ch_hw.shape
    return mask_1ch_hw.view(B, H * W)


def _mask_to_BT(w):
    w = _require_tensor(w, "w")
    if w is None:
        return None

    if w.dim() == 2:
        return w
    if w.dim() == 3:
        B, H, W = w.shape
        return w.view(B, H * W)
    if w.dim() == 4:
        if w.size(1) != 1:
            w = w.mean(1, keepdim=True)
        B, _, H, W = w.shape
        return w.view(B, H * W)

    raise ValueError(f"w must be [B,T] or spatial mask, got {tuple(w.shape)}")


def align_mask_to_T(w, target_T):
    if w is None:
        return None
    w = _mask_to_BT(w)
    B, Tw = w.shape

    if Tw == target_T:
        return w.contiguous()
    if Tw > target_T:
        return w[:, :target_T].contiguous()

    pad = w.new_zeros((B, target_T - Tw))
    return torch.cat([w, pad], dim=1).contiguous()


def next_pred_loss(z, z_hat, w=None, mode="l1"):
    if z.dim() != 3 or z_hat.dim() != 3:
        raise ValueError(f"z/z_hat must be [B,T,D], got {tuple(z.shape)} {tuple(z_hat.shape)}")
    if z.shape != z_hat.shape:
        raise ValueError(f"Shape mismatch: z={tuple(z.shape)} vs z_hat={tuple(z_hat.shape)}")

    if mode == "l1":
        per = (z_hat - z).abs().mean(-1)
    elif mode == "l2":
        per = ((z_hat - z) ** 2).mean(-1).sqrt()
    else:
        raise ValueError("mode must be 'l1' or 'l2'")

    if w is None:
        return per.mean()

    w = align_mask_to_T(w, per.shape[1])
    return (per * w).sum() / (w.sum() + 1e-6)


def diff_loss_1d(z, norm_mask=None, alpha=0.3, beta=1.0, mode="l1"):
    """
    1차 차분(Δz): 인접 패치 임베딩 변화량을 제약하여 급격한 변동을 억제 (smoothness of variation)
    2차 차분(Δ²z): 변화량의 변화(곡률)를 제약하여 임베딩 궤적의 급격한 꺾임을 억제 (curvature regularization)

    최종 손실:  alpha * L(Δz) + beta * L(Δ²z)
    """
    # 차분 로스 사용 확인 (첫 호출 시에만 출력)
    if not hasattr(diff_loss_1d, '_logged'):
        import sys
        print(f"[차분 로스 검증] diff_loss_1d 함수 호출됨 - alpha={alpha:.3f}, beta={beta:.3f}, mode={mode}", file=sys.stderr, flush=True)
        diff_loss_1d._logged = True
    
    if z.dim() != 3:
        raise ValueError(f"z must be [B,T,D], got {tuple(z.shape)}")
    B, T, D = z.shape
    if T < 3:
        return z.new_tensor(0.0)

    # 1차 차분: v_t = z_t - z_{t-1}
    v = z[:, 1:, :] - z[:, :-1, :]  # [B,T-1,D]

    # 2차 차분: a_t = v_t - v_{t-1} = z_t - 2 z_{t-1} + z_{t-2}
    a = v[:, 1:, :] - v[:, :-1, :]  # [B,T-2,D]

    if mode == "l1":
        d1 = v.abs().mean(-1)       # [B,T-1]
        d2 = a.abs().mean(-1)       # [B,T-2]
    elif mode == "l2":
        d1 = (v ** 2).mean(-1).sqrt()
        d2 = (a ** 2).mean(-1).sqrt()
    else:
        raise ValueError("mode must be 'l1' or 'l2'")

    if norm_mask is None:
        return float(alpha) * d1.mean() + float(beta) * d2.mean()

    nm = align_mask_to_T(norm_mask, T)  # [B,T]
    w1 = nm[:, 1:]                      # [B,T-1] (Δz 가중치)
    w2 = nm[:, 2:]                      # [B,T-2] (Δ²z 가중치)

    l1 = (d1 * w1).sum() / (w1.sum() + 1e-6)
    l2 = (d2 * w2).sum() / (w2.sum() + 1e-6)
    return float(alpha) * l1 + float(beta) * l2


def diff_loss_2d(x, alpha=0.05, beta=0.1, mode="l1"):
    """
    Spatial continuity loss for feature maps.

    1st-order terms penalize abrupt horizontal/vertical feature changes.
    2nd-order terms penalize abrupt curvature while preserving broad structure.
    """
    if x.dim() != 4:
        raise ValueError(f"x must be [B,C,H,W], got {tuple(x.shape)}")

    _, _, H, W = x.shape

    def reduce(t):
        if mode == "l1":
            return t.abs().mean()
        if mode == "l2":
            return (t ** 2).mean().sqrt()
        raise ValueError("mode must be 'l1' or 'l2'")

    first_terms = []
    second_terms = []

    if W > 1:
        first_terms.append(reduce(x[:, :, :, 1:] - x[:, :, :, :-1]))
    if H > 1:
        first_terms.append(reduce(x[:, :, 1:, :] - x[:, :, :-1, :]))

    if W > 2:
        second_terms.append(reduce(x[:, :, :, 2:] - 2 * x[:, :, :, 1:-1] + x[:, :, :, :-2]))
    if H > 2:
        second_terms.append(reduce(x[:, :, 2:, :] - 2 * x[:, :, 1:-1, :] + x[:, :, :-2, :]))

    zero = x.new_tensor(0.0)
    first = sum(first_terms) / len(first_terms) if first_terms else zero
    second = sum(second_terms) / len(second_terms) if second_terms else zero
    return float(alpha) * first + float(beta) * second


class SeqProjector(nn.Module):
    def __init__(self, in_ch, d=256):
        super().__init__()
        self.proj = nn.Conv2d(int(in_ch), int(d), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"feature must be [B,C,H,W], got {tuple(x.shape)}")
        x = self.proj(x)
        B, D, H, W = x.shape
        z = x.view(B, D, H * W).transpose(1, 2).contiguous()
        return z, (H, W)


class CausalPredictor(nn.Module):
    def __init__(self, d=256, k=3):
        super().__init__()
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.conv = nn.Conv1d(int(d), int(d), kernel_size=k, padding=0, bias=True)

    def forward(self, z):
        x = z.transpose(1, 2)
        x = F.pad(x, (self.k - 1, 0))
        y = self.conv(x)
        return y.transpose(1, 2).contiguous()


class ContinuityHead(nn.Module):
    def __init__(
        self,
        in_ch: int = None,
        ch_p3: int = None,
        ch_p4: int = None,
        in_ch_p3: int = None,
        in_ch_p4: int = None,
        d: int = 256,
        k: int = 3,
        alpha: float = 0.3,
        beta: float = 1.0,
        lambda_p3: float = 1.0,
        lambda_p4: float = 0.3,
        mode: str = "l1",
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.single_mode = in_ch is not None

        if self.single_mode:
            self.proj = SeqProjector(int(in_ch), int(d))
            self.pred = CausalPredictor(int(d), int(k))
            return

        if ch_p3 is None:
            ch_p3 = in_ch_p3
        if ch_p4 is None:
            ch_p4 = in_ch_p4
        if ch_p3 is None or ch_p4 is None:
            raise ValueError(
                "ContinuityHead: provide `in_ch` for trainer mode, "
                "or provide (ch_p3,ch_p4) for multi-scale mode."
            )

        self.p3_proj = SeqProjector(int(ch_p3), int(d))
        self.p4_proj = SeqProjector(int(ch_p4), int(d))
        self.l3 = float(lambda_p3)
        self.l4 = float(lambda_p4)

    def forward(self, x, p4=None, norm_mask=None):
        mask_1ch = _to_1ch_mask(norm_mask)

        if self.single_mode:
            z, (H, W) = self.proj(x)
            z_hat = self.pred(z)

            z1 = z[:, 1:, :].contiguous()
            z1_hat = z_hat[:, :-1, :].contiguous()

            w1 = None
            if mask_1ch is not None:
                m = resize_mask_to_feature(mask_1ch, H, W)
                m = flatten_mask_hw(m)
                w1 = m[:, 1:].contiguous()

            return z1, z1_hat, w1

        if p4 is None:
            raise ValueError("ContinuityHead multi-scale mode requires forward(p3, p4, ...).")

        z3, (H3, W3) = self.p3_proj(x)
        z4, (H4, W4) = self.p4_proj(p4)

        m3 = None
        m4 = None
        if mask_1ch is not None:
            m3 = flatten_mask_hw(resize_mask_to_feature(mask_1ch, H3, W3))
            m4 = flatten_mask_hw(resize_mask_to_feature(mask_1ch, H4, W4))

        loss3 = diff_loss_1d(z3, m3, alpha=self.alpha, beta=self.beta, mode=self.mode)
        loss4 = diff_loss_1d(z4, m4, alpha=self.alpha, beta=self.beta, mode=self.mode)
        return self.l3 * loss3 + self.l4 * loss4


class ContinuityHead2D(nn.Module):
    def __init__(self, in_ch: int, d: int = 256, k: int = 3, mode: str = "l1", **kwargs):
        super().__init__()
        self.mode = mode
        self.proj = SeqProjector(int(in_ch), int(d))
        self.pred = CausalPredictor(int(d), int(k))

    def forward(self, x, norm_mask=None):
        mask_1ch = _to_1ch_mask(norm_mask)

        z, (H, W) = self.proj(x)
        z_hat = self.pred(z)

        z1 = z[:, 1:, :].contiguous()
        z1_hat = z_hat[:, :-1, :].contiguous()

        w1 = None
        if mask_1ch is not None:
            m = resize_mask_to_feature(mask_1ch, H, W)
            m = flatten_mask_hw(m)
            w1 = m[:, 1:].contiguous()

        return z1, z1_hat, w1


def _extract_tensor_any(feats):
    if torch.is_tensor(feats):
        return feats
    if isinstance(feats, (list, tuple)):
        return feats[0]
    if isinstance(feats, dict):
        x = feats.get("p3", feats.get("P3", None))
        if x is None:
            x = feats.get("p4", feats.get("P4", None))
        if x is None:
            x = next(iter(feats.values()))
        return x
    raise TypeError("feats must be tensor or list/tuple/dict")


def _extract_p3_p4(feats):
    if isinstance(feats, dict):
        p3 = feats.get("p3", feats.get("P3", None))
        p4 = feats.get("p4", feats.get("P4", None))
        if p3 is None or p4 is None:
            raise ValueError("feats dict must contain keys 'p3'/'p4' (or 'P3'/'P4').")
        return p3, p4
    if isinstance(feats, (list, tuple)):
        if len(feats) < 2:
            raise ValueError("feats must have at least [p3, p4].")
        return feats[0], feats[1]
    raise TypeError("feats must be dict or list/tuple")


def npp_loss(*args, **kwargs):
    mode = kwargs.pop("mode", "l1")

    if len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
        z1 = args[0]
        z1_hat = args[1]
        w1 = args[2] if len(args) >= 3 else None
        
        # alpha, beta가 kwargs에 있으면 차분 로스도 계산
        alpha = kwargs.pop("alpha", None)
        beta = kwargs.pop("beta", None)
        
        if alpha is not None and beta is not None:
            # 차분 로스 사용 확인 (첫 호출 시에만 출력)
            if not hasattr(npp_loss, '_diff_logged'):
                import sys
                print(f"[차분 로스 검증] npp_loss에서 차분 로스 사용 - alpha={alpha:.3f}, beta={beta:.3f}", file=sys.stderr, flush=True)
                npp_loss._diff_logged = True
            
            # next_pred_loss와 diff_loss_1d 모두 계산
            l_next = next_pred_loss(z1, z1_hat, w=w1, mode=mode)
            l_diff = diff_loss_1d(z1, w1, alpha=alpha, beta=beta, mode=mode)
            return l_next + l_diff  # 차분 로스 포함
        
        # alpha, beta가 없으면 next_pred_loss만 사용
        if not hasattr(npp_loss, '_next_only_logged'):
            import sys
            print(f"[차분 로스 검증] npp_loss에서 next_pred_loss만 사용 (차분 로스 미사용)", file=sys.stderr, flush=True)
            npp_loss._next_only_logged = True
        
        return next_pred_loss(z1, z1_hat, w=w1, mode=mode)

    if len(args) < 2:
        raise TypeError("npp_loss expects either (z1,z1_hat,...) or (continuity_head, feats, ...).")

    continuity_head = args[0]
    feats = args[1]

    norm_mask = kwargs.pop("norm_mask", None)
    _ = kwargs.pop("use_p5", False)

    loss_type = kwargs.pop("loss_type", "next+diff")
    next_w = float(kwargs.pop("next_w", 1.0))
    diff_w = float(kwargs.pop("diff_w", 1.0))
    alpha = float(kwargs.pop("alpha", 0.3))
    beta = float(kwargs.pop("beta", 1.0))

    if getattr(continuity_head, "single_mode", False):
        x = _extract_tensor_any(feats)
        z1, z1_hat, w1 = continuity_head(x, norm_mask=norm_mask)

        l_next = next_pred_loss(z1, z1_hat, w=w1, mode=mode)
        l_diff = diff_loss_1d(z1, w1, alpha=alpha, beta=beta, mode=mode) if diff_w > 0 else z1.new_tensor(0.0)

        if loss_type == "next":
            return l_next
        if loss_type == "diff":
            return l_diff
        return next_w * l_next + diff_w * l_diff

    p3, p4 = _extract_p3_p4(feats)
    return continuity_head(p3, p4, norm_mask=norm_mask)


def npp_loss_2d(*args, **kwargs):
    return npp_loss(*args, **kwargs)
