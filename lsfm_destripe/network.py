import haiku as hk
import numpy as np
import jax.numpy as jnp
import jax
import math
from scipy import ndimage
from jax import random
from jax import jit
from functools import partial
import scipy
import torch.fft as fft
import torch
from lsfm_destripe.utils import crop_center
from lsfm_destripe.utils_jax import generate_mapping_coordinates


class Cmplx_Xavier_Init(hk.initializers.Initializer):
    def __init__(
        self,
        input_units,
        output_units,
    ):
        self.n_in, self.n_out = input_units, output_units

    def __call__(
        self,
        shape,
        dtype,
    ):
        magnitudes = jnp.ones(shape) / self.n_in
        phases = jax.random.uniform(
            hk.next_rng_key(),
            shape,
            dtype="float32",
            minval=-np.pi,
            maxval=np.pi,
        )
        return magnitudes * jnp.exp(1.0j * phases)


class CmplxRndUniform(hk.initializers.Initializer):
    def __init__(
        self,
        minval=0,
        maxval=1.0,
    ):
        self.minval, self.maxval = minval, maxval

    def __call__(
        self,
        shape,
        dtype,
    ):
        real_part = jax.random.uniform(
            hk.next_rng_key(),
            shape,
            dtype="float32",
            minval=self.minval,
            maxval=self.maxval,
        )
        imag_part = jax.random.uniform(
            hk.next_rng_key(),
            shape,
            dtype="float32",
            minval=self.minval,
            maxval=self.maxval,
        )
        return jax.lax.complex(real_part, imag_part)


class CLinear(hk.Module):
    def __init__(
        self,
        output_size,
        name=None,
    ):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(
        self,
        inputs,
    ):
        input_size = inputs.shape[-1]
        dtype = inputs.dtype
        w = hk.get_parameter(
            "w",
            [input_size, self.output_size],
            dtype,
            init=Cmplx_Xavier_Init(input_size, self.output_size),
        )
        b = hk.get_parameter(
            "b",
            [self.output_size],
            dtype,
            init=CmplxRndUniform(minval=-0.001, maxval=0.001),
        )
        return jnp.dot(inputs, w) + b


def complexReLU(x):
    return jax.lax.complex(jax.nn.elu(x.real), jax.nn.elu(x.imag))


class ResLearning(hk.Module):
    def __init__(
        self,
        outc,
    ):
        super().__init__()
        self.outc = outc

    def __call__(
        self,
        x,
    ):
        inx = CLinear(self.outc)(complexReLU(CLinear(self.outc)(x)))
        return complexReLU(inx + CLinear(self.outc)(x))


class dual_view_fusion:
    def __init__(
        self,
    ):
        pass

    def __call__(
        self,
        x,
        boundary,
    ):
        return (x * boundary).sum(1, keepdims=True)


class identical_func:
    def __init__(
        self,
    ):
        pass

    def __call__(self, x, boundary):
        return x


class gnn(hk.Module):
    def __init__(
        self,
        NI,
        hier_mask,
        hier_ind,
        inc,
    ):
        super().__init__()
        self.NI, self.hier_mask, self.hier_ind = NI, hier_mask, hier_ind
        self.inc = inc

    def __call__(
        self,
        Xf,
    ):
        # return Xf
        w = hk.get_parameter(
            "w",
            (self.NI.shape[0], self.NI.shape[1]),
            jnp.complex64,
            init=Cmplx_Xavier_Init(self.NI.shape[0], self.NI.shape[1]),
        )
        Xfcx = jnp.sum(jnp.einsum("knbc,kn->knbc", Xf[self.NI, :], w), 0)  # (M*N, c)
        Xf_tvx = jnp.concatenate((Xfcx, Xf[self.hier_mask, :]), 0)[
            self.hier_ind, :
        ].reshape(
            1, -1, 1, self.inc
        )  # (A, M*N, 2, c)
        return Xf_tvx


class gnn_dual(hk.Module):
    def __init__(
        self,
        NI,
        hier_mask,
        hier_ind,
        inc,
    ):
        super().__init__()
        self.gnn_0 = gnn(NI[0], hier_mask[0], hier_ind[0], inc)
        self.gnn_1 = gnn(NI[1], hier_mask[1], hier_ind[1], inc)

    def __call__(
        self,
        Xf,
    ):
        Xf_0, Xf_1 = jnp.split(Xf, indices_or_sections=2, axis=-2)
        return [self.gnn_0(Xf_0), self.gnn_1(Xf_1)]


class tv_uint(hk.Module):
    def __init__(
        self,
        TVfftx,
        TVffty,
        inverseTVfftx,
        inverseTVffty,
        eigDtD,
        edgeProcess,
        latentProcess,
    ):
        super().__init__()
        self.TVfftx, self.TVffty = TVfftx, TVffty
        self.inverseTVfftx, self.inverseTVffty = inverseTVfftx, inverseTVffty
        self.eigDtD = eigDtD
        self.edgeProcess, self.latentProcess = edgeProcess, latentProcess

    def __call__(self, Xf_tvx, Xf):
        Xf_tvx, Xf_tvy = Xf_tvx * self.TVfftx, Xf * self.TVffty
        X_fourier = []
        for (
            x,
            y,
            inverseTVfftx,
            inverseTVffty,
            eigDtD,
            edgeProcess,
            latentProcess,
        ) in zip(
            Xf_tvx,
            Xf_tvy,
            self.inverseTVfftx,
            self.inverseTVffty,
            self.eigDtD,
            self.edgeProcess,
            self.latentProcess,
        ):
            X_fourier.append(
                complexReLU(
                    latentProcess(
                        complexReLU(edgeProcess(x) * inverseTVfftx)
                        + complexReLU(edgeProcess(y) * inverseTVffty)
                    )
                    / eigDtD
                )
            )
        return sum(X_fourier)  # self.merge(jnp.concatenate(X_fourier, -1))


class tv_uint_dual(hk.Module):
    def __init__(
        self,
        TVfftx,
        TVffty,
        inverseTVfftx,
        inverseTVffty,
        eigDtD,
        edgeProcess,
        latentProcess,
    ):
        super().__init__()
        self.tv_uint0 = tv_uint(
            TVfftx[0],
            TVffty[0],
            inverseTVfftx[0],
            inverseTVffty[0],
            eigDtD[0],
            edgeProcess[0],
            latentProcess[0],
        )
        self.tv_uint1 = tv_uint(
            TVfftx[1],
            TVffty[1],
            inverseTVfftx[1],
            inverseTVffty[1],
            eigDtD[1],
            edgeProcess[1],
            latentProcess[1],
        )

    def __call__(self, Xf_tvx, Xf):
        X_fourier0 = self.tv_uint0(Xf_tvx[0], Xf[:, 0:1, :])
        X_fourier1 = self.tv_uint1(Xf_tvx[1], Xf[:, 1:2, :])
        return jnp.concatenate((X_fourier0, X_fourier1), -2)


class non_positive_unit:
    def __init__(
        self,
    ):
        pass

    def __call__(self, x, target):
        return jnp.abs(x - target) + target


class DeStripeModel(hk.Module):
    def __init__(
        self,
        Angle,
        hier_mask,
        hier_ind,
        NI,
        m_l,
        n_l,
        m_h,
        n_h,
        KS,
        r,
        Angle_X1=None,
        Angle_X2=None,
        inc=16,
        GFr=49,
        viewnum=1,
        device="gpu",
        non_positive=False,
    ):
        super().__init__()
        self.NI, self.hier_mask, self.hier_ind, self.inc = NI, hier_mask, hier_ind, inc

        self.m_l, self.n_l = m_l, n_l
        self.m_h, self.n_h = m_h, n_h
        self.Angle = Angle

        gx_0 = jnp.fft.fftshift(
            jnp.fft.fft2(jnp.array([[1, -1]], dtype=jnp.float32), (self.m_l, self.n_l))
        )
        gy_0 = jnp.fft.fftshift(
            jnp.fft.fft2(
                jnp.array([[1], [-1]], dtype=jnp.float32), (self.m_l, self.n_l)
            )
        )

        self.TVfftx = []
        self.inverseTVfftx = []
        self.TVffty = []
        self.inverseTVffty = []
        for i, A in enumerate(Angle):
            trans_matrix = generate_mapping_coordinates(
                np.rad2deg(np.arctan(r * np.tan(np.deg2rad(-A)))),
                gx_0.shape[0],
                gx_0.shape[1],
                reshape=False,
            )
            self.fftnt(
                jax.scipy.ndimage.map_coordinates(
                    gx_0[None, None], trans_matrix, 1, mode="nearest"
                )[0, 0],
                self.m_l,
                self.n_l,
            )
            self.fftn(
                jax.scipy.ndimage.map_coordinates(
                    gy_0[None, None], trans_matrix, 1, mode="nearest"
                )[0, 0],
                self.m_l,
                self.n_l,
            )
        self.TVfftx = jnp.concatenate(self.TVfftx, 0)
        self.inverseTVfftx = jnp.concatenate(self.inverseTVfftx, 0)
        self.TVffty = jnp.concatenate(self.TVffty, 0)
        self.inverseTVffty = jnp.concatenate(self.inverseTVffty, 0)

        self.eigDtD = jnp.power(jnp.abs(self.TVfftx), 2) + jnp.power(
            jnp.abs(self.TVffty), 2
        )
        self.TVfftx, self.inverseTVfftx = (
            self.TVfftx[..., None],
            self.inverseTVfftx[..., None],
        )
        self.TVffty, self.inverseTVffty = (
            self.TVffty[..., None],
            self.inverseTVffty[..., None],
        )
        self.eigDtD = self.eigDtD[..., None]

        self.p = ResLearning(inc)
        self.edgeProcess = []
        for _ in Angle:
            self.edgeProcess.append(
                hk.Sequential([CLinear(inc), complexReLU, CLinear(inc)])
            )
        self.latentProcess = []
        for _ in Angle:
            self.latentProcess.append(
                hk.Sequential([CLinear(inc), complexReLU, CLinear(inc)])
            )
        self.basep = hk.Sequential(
            [
                hk.Linear(inc),
                jax.nn.relu,
                hk.Linear(inc),
                jax.nn.relu,
                hk.Linear(1),
            ]
        )
        self.basep2 = hk.Sequential(
            [
                hk.Linear(inc),
                jax.nn.relu,
                hk.Linear(inc),
                jax.nn.relu,
                hk.Linear(1),
            ]
        )
        self.ainput = jnp.ones((viewnum, 1))
        self.merge = hk.Sequential(
            [
                CLinear(inc),
                complexReLU,
                CLinear(inc),
                complexReLU,
                CLinear(1),
            ]
        )
        self.viewnum = viewnum

        self.fuse = dual_view_fusion() if viewnum == 2 else identical_func()
        self.gnn = (
            gnn_dual(self.NI, self.hier_mask, self.hier_ind, inc)
            if viewnum == 2
            else gnn(self.NI, self.hier_mask, self.hier_ind, inc)
        )

        self.tv_uint = (
            tv_uint_dual(
                [
                    self.TVfftx[np.isin(Angle, Angle_X1)],
                    self.TVfftx[np.isin(Angle, Angle_X2)],
                ],
                [
                    self.TVffty[np.isin(Angle, Angle_X1)],
                    self.TVffty[np.isin(Angle, Angle_X2)],
                ],
                [
                    self.inverseTVfftx[np.isin(Angle, Angle_X1)],
                    self.inverseTVfftx[np.isin(Angle, Angle_X2)],
                ],
                [
                    self.inverseTVffty[np.isin(Angle, Angle_X1)],
                    self.inverseTVffty[np.isin(Angle, Angle_X2)],
                ],
                [
                    self.eigDtD[np.isin(Angle, Angle_X1)],
                    self.eigDtD[np.isin(Angle, Angle_X2)],
                ],
                [
                    [
                        self.edgeProcess[i]
                        for i, j in enumerate(np.isin(Angle, Angle_X1))
                        if j
                    ],
                    [
                        self.edgeProcess[i]
                        for i, j in enumerate(np.isin(Angle, Angle_X2))
                        if j
                    ],
                ],
                [
                    [
                        self.latentProcess[i]
                        for i, j in enumerate(np.isin(Angle, Angle_X1))
                        if j
                    ],
                    [
                        self.latentProcess[i]
                        for i, j in enumerate(np.isin(Angle, Angle_X2))
                        if j
                    ],
                ],
            )
            if viewnum == 2
            else tv_uint(
                self.TVfftx,
                self.TVffty,
                self.inverseTVfftx,
                self.inverseTVffty,
                self.eigDtD,
                self.edgeProcess,
                self.latentProcess,
            )
        )
        self.non_positive_unit = (
            non_positive_unit() if non_positive else identical_func()
        )

    def fftnt(self, x, row, col):
        y = x.reshape(-1)[: col * row // 2][None, :, None]
        self.TVfftx.append(y)
        self.inverseTVfftx.append(jnp.conj(y))

    def fftn(self, x, row, col):
        y = x.reshape(-1)[: col * row // 2][None, :, None]
        self.TVffty.append(y)
        self.inverseTVffty.append(jnp.conj(y))

    def fourierResult(self, z, aver):
        return jnp.abs(
            jnp.fft.ifft2(
                jnp.fft.ifftshift(
                    jnp.concatenate(
                        (z, aver, jnp.conj(jnp.flip(z, -2))),
                        -2,
                    )
                    .reshape(1, self.m_l, -1, self.viewnum)
                    .transpose(0, 3, 1, 2),
                    axes=(-2, -1),
                )
            )
        )

    def __call__(self, aver, Xf, target, boundary, target_hr):
        Xf = self.p(Xf)  # (M*N, 2,)
        Xf_tvx = self.gnn(Xf)
        X_fourier = self.merge(self.tv_uint(Xf_tvx, Xf))
        outputGNNraw = self.fourierResult(X_fourier[..., 0], aver)
        alpha = hk.get_parameter(
            "alpha",
            (1, self.viewnum, 1, 1),
            jnp.float32,
            init=jnp.ones,
        )
        outputGNNraw = self.fuse(outputGNNraw + alpha, boundary)
        outputGNNraw = self.non_positive_unit(outputGNNraw, target)
        return outputGNNraw
