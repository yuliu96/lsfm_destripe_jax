import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers
from functools import partial
from jax import jit, value_and_grad
import math
import SimpleITK as sitk
from typing import List


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
