from typing import List
import numpy as np
import scipy
import haiku as hk


def transform_cmplx_model(
    model,
    backend,
    device,
    **model_kwargs,
):
    def forward_pass(**x):
        net = model(**model_kwargs)
        return net(**x)

    if backend == "jax":
        network = hk.without_apply_rng(hk.transform(forward_pass))
    else:
        network = model(**model_kwargs).to(device)
    return network


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
    gf_kernel_size_in_y: int = 3,
    gf_mode: int = 1,
    backend: str = "jax",
):
    kwargs = locals()
    return kwargs
