import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers
from functools import partial
from jax import jit, value_and_grad


@optimizers.optimizer
def cADAM(step_size, b1=0.9, b2=0.999, eps=1e-8):
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


def transform_cmplx_haiku_model(model, **model_kwargs):
    def forward_pass(**x):
        net = model(**model_kwargs)
        return net(**x)

    network = hk.without_apply_rng(hk.transform(forward_pass))
    return network


def initialize_cmplx_haiku_model(network, key, dummy_input):
    net_params = network.init(key, **dummy_input)
    return net_params


class update_jax:
    def __init__(self, network, Loss, learning_rate):
        self.opt_init, self.opt_update, self.get_params = cADAM(learning_rate)
        self.loss = Loss
        self._network = network

    @partial(jit, static_argnums=(0))
    def __call__(
        self,
        step,
        params,
        opt_state,
        x,
        xf,
        boundary,
        y,
        smoothedTarget,
        map,
        hy,
    ):
        (l, (A, B, C)), grads = value_and_grad(self.loss, has_aux=True)(
            params,
            self._network,
            {"X": x, "Xf": xf, "boundary": boundary, "target": y},
            y,
            smoothedTarget,
            map,
            hy,
        )
        grads = jax.tree_map(jnp.conjugate, grads)
        opt_state = self.opt_update(step, grads, opt_state)
        return self.get_params(opt_state), opt_state, A, B, C
