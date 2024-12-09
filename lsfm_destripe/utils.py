from typing import List
import numpy as np
import scipy


def crop_center(
    img,
    cropy,
    cropx,
):
    y, x = img.shape[-2:]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[..., starty : starty + cropy, startx : startx + cropx]


def global_correction(
    mean,
    result,
):
    means = scipy.signal.savgol_filter(mean, min(21, len(mean)), 1)
    MIN, MAX = result.min(), result.max()
    result = result - mean[:, None, None] + means[:, None, None]
    result = (result - result.min()) / (result.max() - result.min()) * (MAX - MIN) + MIN
    return np.clip(result, 0, 65535).astype(np.uint16)


def destripe_train_params(
    resample_ratio: int = 3,
    gf_kernel_size: int = 29,
    hessian_kernel_sigma: float = 0.5,
    lambda_masking_mse: int = 2,
    lambda_tv: float = 1,
    lambda_hessian: float = 1,
    inc: int = 16,
    n_epochs: int = 300,
    wedge_degree: float = 29,
    n_neighbors: int = 16,
    fusion_kernel_size: int = 49,
    require_global_correction: bool = True,
    fast_mode: bool = False,
    fidelity_first: bool = False,
):
    kwargs = locals()
    return kwargs
