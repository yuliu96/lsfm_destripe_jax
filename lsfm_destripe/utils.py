from typing import List
import numpy as np
import math
import scipy
import jax
import jax.numpy as jnp
import copy
from lsfm_destripe.utils_jax import generate_mapping_coordinates


def crop_center(
    img,
    cropy,
    cropx,
):
    y, x = img.shape[-2:]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[..., starty : starty + cropy, startx : startx + cropx]


def NeighborSampling(
    m,
    n,
    k_neighbor=16,
):
    """
    Do neighbor sampling

    Parameters:
    ---------------------
    m: int
        size of neighbor along X dim
    n: int
        size of neighbor along Y dim
    k_neigher: int, data range [1, 32], 16 by default
        number of neighboring points
    """
    width = 11
    NI = jnp.zeros((m * n, k_neighbor), dtype=jnp.int32)
    key = jax.random.key(0)
    grid_x, grid_y = jnp.meshgrid(
        jnp.linspace(1, m, m), jnp.linspace(1, n, n), indexing="ij"
    )
    grid_x, grid_y = grid_x - math.floor(m / 2) - 1, grid_y - math.floor(n / 2) - 1
    grid_x, grid_y = grid_x.reshape(-1) ** 2, grid_y.reshape(-1) ** 2

    iter_num = jnp.sqrt((grid_x + grid_y).max()) // width + 1

    mask_outer = (grid_x + grid_y) < (width * jnp.arange(1, iter_num + 1)[:, None]) ** 2
    mask_inner = (grid_x + grid_y) >= (width * jnp.arange(0, iter_num)[:, None]) ** 2
    mask = mask_outer * mask_inner
    ind = jnp.where(mask)
    _, counts = jnp.unique(ind[0], return_counts=True)
    counts_cumsum = jnp.cumsum(counts)

    low = jnp.concatenate(
        (jnp.array([0]), counts_cumsum[:-1]),
    )

    low = low.repeat(counts)
    high = counts_cumsum
    high = high.repeat(counts)
    indc = jax.random.randint(key, (k_neighbor, len(low)), low, high).T
    NI = NI.at[ind[1]].set(ind[1][indc])
    zero_freq = (m * n) // 2
    NI = NI[:zero_freq, :]
    NI = NI.at[NI > zero_freq].set(2 * zero_freq - NI[NI > zero_freq])
    return jnp.concatenate(
        (jnp.linspace(0, NI.shape[0] - 1, NI.shape[0])[:, jnp.newaxis], NI),
        axis=1,
    ).astype(jnp.int32)


def WedgeMask(
    md,
    nd,
    Angle,
    deg,
    fast_mode,
):
    """
    Add docstring here
    """
    md_o, nd_o = copy.deepcopy(md), copy.deepcopy(nd)
    md = max(md_o, nd_o)
    nd = max(md_o, nd_o)

    Xv, Yv = jnp.meshgrid(jnp.linspace(0, nd, nd + 1), jnp.linspace(0, md, md + 1))
    tmp = jnp.arctan2(Xv, Yv)
    tmp = jnp.hstack((jnp.flip(tmp[:, 1:], 1), tmp))
    tmp = jnp.vstack((jnp.flip(tmp[1:, :], 0), tmp))
    if Angle != 0:
        rotate_mask = generate_mapping_coordinates(
            -Angle,
            tmp.shape[0],
            tmp.shape[1],
            False,
        )
        tmp = jax.scipy.ndimage.map_coordinates(
            tmp[None, None],
            rotate_mask,
            0,
            mode="nearest",
        )[0, 0]

    a = crop_center(tmp, md, nd)

    tmp = Xv**2 + Yv**2
    tmp = jnp.hstack((jnp.flip(tmp[:, 1:], 1), tmp))
    tmp = jnp.vstack((jnp.flip(tmp[1:, :], 0), tmp))
    b = tmp[md - md // 2 : md + md // 2 + 1, nd - nd // 2 : nd + nd // 2 + 1]
    b = (
        jnp.abs(jnp.arange(nd) - nd // 2)[None, :] > nd // 4 * jnp.ones(md)[:, None]
    ).astype(jnp.int32)
    if fast_mode:
        return crop_center(
            (
                ((a < math.pi / 180 * (90 - deg)).astype(jnp.int32) + b)
                * (b > 1024).astype(jnp.int32)
            )
            != 0,
            md_o,
            nd_o,
        )
    else:
        return crop_center(
            (
                ((a < math.pi / 180 * (90 - deg)).astype(jnp.int32))
                * (b > 1024).astype(jnp.int32)
            )
            != 0,
            md_o,
            nd_o,
        )


def prepare_aux(
    md: int,
    nd: int,
    fast_mode: bool,
    is_vertical: bool,
    angleOffset: List[float] = None,
    deg: float = 29,
    Nneighbors: int = 16,
    NI_all=None,
):
    """
    the function preparing auxillary variables for training based on image shape

    Parameters:
    ------------
    md: int
        sampling nbr size along Y
    nd: int
        sampling nbr size along X
    is_verticel: book
        if the stripes are vertical
    angleOffset: TODO
        TODO
    deg: float
        TODO
    Nneighbors: int
        TODO

    Returns:
    -------------
    NI: ndarray
        TODO
    hier_mask: ndarray
        TODO
    hier_ind: ndarray
        TODO
    """
    if not is_vertical:
        (nd, md) = (md, nd)

    angleMask = jnp.ones((md, nd), dtype=np.int32)
    for angle in angleOffset:
        angleMask = angleMask * WedgeMask(
            md,
            nd,
            Angle=angle,
            deg=deg,
            fast_mode=fast_mode,
        )
    angleMask = angleMask[None]
    angleMask = angleMask.reshape(angleMask.shape[0], -1)[:, : md * nd // 2]
    hier_mask = jnp.where(angleMask == 1)[1]  ##(3, N)

    hier_ind = jnp.argsort(
        jnp.concatenate(
            [jnp.where(angleMask.reshape(-1) == index)[0] for index in range(2)]
        )
    )
    if NI_all is None:
        NI_all = NeighborSampling(md, nd, k_neighbor=Nneighbors)
    NI = jnp.concatenate(
        [NI_all[angle_mask == 0, :].T for angle_mask in angleMask], 1
    )  # 1 : Nneighbors + 1
    return hier_mask, hier_ind, NI, NI_all


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
    sampling_in_MSEloss: int = 2,
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
