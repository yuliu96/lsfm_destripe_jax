import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers
from functools import partial
from jax import jit, value_and_grad
import math
import SimpleITK as sitk
import copy
from typing import List
from lsfm_destripe.utils import crop_center


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
    bb = (
        jnp.abs(jnp.arange(nd) - nd // 2)[None, :] > nd // 4 * jnp.ones(md)[:, None]
    ).astype(jnp.int32)
    if fast_mode:
        return crop_center(
            (
                ((a < math.pi / 180 * (90 - deg)).astype(jnp.int32) + bb)
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
    backend="jax",
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


def generate_mask_dict(
    dualtargetd,
    fusion_mask,
    Dx,
    Dy,
    DGaussxx,
    DGaussyy,
    p_tv,
    p_hessian,
    train_params,
    sample_params,
    backend,
):
    r = train_params["max_pool_kernel_size"]
    md, nd = dualtargetd.shape[-2:]
    dualtargetd_pad = jnp.pad(
        dualtargetd, ((0, 0), (0, 0), (0, 0), (r // 2, r // 2)), "reflect"
    )
    ind = jax.lax.conv_general_dilated_patches(
        dualtargetd_pad, (1, r), (1, 1), "VALID"
    ).argmax(axis=1, keepdims=True)
    ind = ind + jnp.arange(nd)[None, None, None, :]
    mask_tv = (
        jnp.arctan2(
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(dualtargetd, p_tv, "reflect"), Dx, (1, 1), "VALID"
                )
            ),
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(dualtargetd, p_tv, "reflect"), Dy, (1, 1), "VALID"
                )
            ),
        )
        ** 12
    )

    mask_hessian = (
        jnp.arctan2(
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(dualtargetd, p_hessian, "reflect"),
                    DGaussxx,
                    (1, 1),
                    "VALID",
                )
            ),
            jnp.abs(
                jax.lax.conv_general_dilated(
                    jnp.pad(dualtargetd, p_hessian, "reflect"),
                    DGaussyy,
                    (1, 1),
                    "VALID",
                )
            ),
        )
        ** 12
    )

    mask_valid = jnp.zeros_like(mask_hessian)
    for i, ao in enumerate(sample_params["angle_offset_individual"]):
        mask_xi_f = np.isin(sample_params["angle_offset"], ao)
        mask_xi = mask_xi_f[:, None].repeat(5, 1).reshape(-1)
        mask_valid += mask_xi[None, :, None, None] * fusion_mask[:, i : i + 1, :, :]
    mask_valid = mask_valid > 0
    mask_tv = mask_tv * mask_valid[:, ::5, :, :]
    mask_hessian = mask_hessian * mask_valid

    mask_tv_f = -hk.max_pool(
        -mask_tv,
        [1, sample_params["r"]],
        [1, sample_params["r"]],
        "VALID",
        channel_axis=1,
    )

    ind_tv_f = jnp.argmax(mask_tv_f, axis=1, keepdims=True)
    mask_tv_f = jnp.max(mask_tv_f, axis=1, keepdims=True)

    ind_tv = jnp.argmax(mask_tv, axis=1, keepdims=True)
    mask_tv = jnp.max(mask_tv, axis=1, keepdims=True)

    mask_hessian_f_split = jnp.split(
        jnp.arange(mask_hessian.shape[1]), len(sample_params["angle_offset"])
    )
    mask_hessian_f = []
    for data_ind in mask_hessian_f_split:
        mask_hessian_f.append(
            jnp.mean(mask_hessian[:, data_ind, :, :], 1, keepdims=True)
        )
    mask_hessian_f = jnp.concatenate(mask_hessian_f, 1)
    mask_hessian_f = -hk.max_pool(
        -mask_hessian_f,
        [1, sample_params["r"]],
        [1, sample_params["r"]],
        "VALID",
        channel_axis=1,
    )

    ind_hessian_f = jnp.argmax(mask_hessian_f, axis=1, keepdims=True)
    mask_hessian_f = jnp.max(mask_hessian_f, axis=1, keepdims=True)

    ind_hessian = jnp.argmax(mask_hessian, axis=1, keepdims=True)
    mask_hessian = jnp.max(mask_hessian, axis=1, keepdims=True)
    mask_dict = {
        "mask_tv": mask_tv,
        "mask_hessian": mask_hessian,
        "mask_hessian_f": mask_hessian_f,  # jnp.where(mask_hessian_f>1, mask_hessian_f, 0),
        "mask_tv_f": mask_tv_f,  # jnp.where(mask_tv_f>1, mask_tv_f, 0),
        "ind": ind,
        "ind_tv": ind_tv,
        "ind_hessian": ind_hessian,
        "ind_hessian_f": ind_hessian_f,
        "ind_tv_f": ind_tv_f,
    }
    return mask_dict


def generate_mapping_matrix(
    angle,
    m,
    n,
):
    affine = sitk.Euler2DTransform()
    affine.SetCenter([m / 2, n / 2])
    affine.SetAngle(angle / 180 * math.pi)
    A = np.array(affine.GetMatrix()).reshape(2, 2)
    c = np.array(affine.GetCenter())
    t = np.array(affine.GetTranslation())
    T = np.eye(3, dtype=np.float32)
    T[0:2, 0:2] = A
    T[0:2, 2] = -np.dot(A, c) + t + c
    return T


def generate_mapping_coordinates(
    angle,
    m,
    n,
    reshape=True,
):
    T = generate_mapping_matrix(angle, m, n)
    id = np.array([[0, 0], [0, n], [m, 0], [m, n]]).T
    if reshape:
        out_bounds = T[:2, :2] @ id
        out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = np.array([m, n])

    out_center = T[:2, :2] @ ((out_plane_shape - 1) / 2)
    in_center = (np.array([m, n]) - 1) / 2
    offset = in_center - out_center
    xx, yy = jnp.meshgrid(
        jnp.linspace(0, out_plane_shape[0] - 1, out_plane_shape[0]),
        jnp.linspace(0, out_plane_shape[1] - 1, out_plane_shape[1]),
    )
    T = jnp.array(T)
    z = (
        jnp.dot(T[:2, :2], jnp.stack((xx, yy)).reshape(2, -1)).reshape(
            2, out_plane_shape[1], out_plane_shape[0]
        )
        + offset[:, None, None]
    )
    z = z.transpose(0, 2, 1)
    z = jnp.concatenate((jnp.zeros_like(z), z))[:, None, None]
    return z


@optimizers.optimizer
def cADAM(
    step_size,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
):
    step_size = optimizers.make_schedule(step_size)

    def init(x0):
        return x0, jnp.zeros_like(x0), jnp.zeros_like(x0)

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m
        v = (1 - b2) * jnp.array(g) * jnp.conjugate(g) + b2 * v
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


def initialize_cmplx_model(
    network,
    key,
    dummy_input,
    backend,
):
    if backend == "jax":
        net_params = network.init(key, **dummy_input)
        return net_params
    else:
        net_params = network.parameters()
        return net_params


class update_jax:
    def __init__(
        self,
        network,
        Loss,
        learning_rate,
    ):
        self.opt_init, self.opt_update, self.get_params = cADAM(learning_rate)
        self.loss = Loss
        self._network = network

    @partial(jit, static_argnums=(0))
    def __call__(
        self,
        step,
        params,
        opt_state,
        aver,
        xf,
        y,
        mask_dict,
        hy,
        targets_f,
        targetd_bilinear,
    ):
        (l, A), grads = value_and_grad(self.loss, has_aux=True)(
            params,
            self._network,
            {
                "aver": aver,
                "Xf": xf,
                "target": y,
                "target_hr": hy,
                "coor": mask_dict["coor"],
            },
            targetd_bilinear,
            mask_dict,
            hy,
            targets_f,
        )
        grads = jax.tree_util.tree_map(jnp.conjugate, grads)
        opt_state = self.opt_update(step, grads, opt_state)
        return l, self.get_params(opt_state), opt_state, A
