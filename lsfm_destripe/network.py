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


class Cmplx_Xavier_Init(hk.initializers.Initializer):
    def __init__(self, input_units, output_units):
        self.n_in, self.n_out = input_units, output_units

    def __call__(self, shape, dtype):
        sigma = 1 / np.sqrt(self.n_in + self.n_out)
        magnitudes = jnp.array(
            np.random.rayleigh(scale=sigma, size=shape),
            dtype="float32",
        )
        phases = jax.random.uniform(
            hk.next_rng_key(),
            shape,
            dtype="float32",
            minval=-np.pi,
            maxval=np.pi,
        )
        return magnitudes * jnp.exp(1.0j * phases)


class CmplxRndUniform(hk.initializers.Initializer):
    def __init__(self, minval=0, maxval=1.0):
        self.minval, self.maxval = minval, maxval

    def __call__(self, shape, dtype):
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
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, inputs):
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
    return jax.lax.complex(jax.nn.relu(x.real), jax.nn.relu(x.imag))


complexReLUJIT = jax.jit(complexReLU)


class ResLearning(hk.Module):
    def __init__(self, outc):
        super().__init__()
        self.outc = outc

    def __call__(self, x):
        inx = hk.Sequential([CLinear(self.outc), complexReLUJIT, CLinear(self.outc)])(x)
        return complexReLUJIT(inx + CLinear(self.outc)(x))


class GuidedFilterJAX(hk.Module):
    def __init__(self, rx, ry, Angle, m=None, n=None, eps=1e-9):
        super().__init__()
        self.eps = eps
        kernelL = []
        self.AngleNum = len(Angle)
        self.Angle = Angle
        for A in Angle:
            kernel = np.zeros((rx * 4 + 1, rx * 4 + 1), dtype=np.float32)
            if ry == 0:
                kernel[:, rx * 2] = 1
            else:
                kernel[:, rx * 2 - ry : rx * 2 + ry + 1] = 1
            kernel = ndimage.rotate(kernel, A, reshape=False, order=2)[
                rx : 3 * rx + 1, rx : 3 * rx + 1
            ]
            r, c = sum(kernel.sum(1) != 0) // 2 * 2, sum(kernel.sum(0) != 0) // 2 * 2
            kernelL.append(
                kernel[rx - r // 2 : rx + r // 2 + 1, rx - c // 2 : rx + c // 2 + 1][
                    None, None
                ]
            )
        c, r = max([k.shape[-2] for k in kernelL]), max([k.shape[-1] for k in kernelL])
        self.kernel = np.zeros((len(Angle), 1, c, r))
        for i, k in enumerate(kernelL):
            self.kernel[
                i,
                :,
                (c - k.shape[-2])
                // 2 : (
                    -(c - k.shape[-2]) // 2 if -(c - k.shape[-2]) // 2 != 0 else None
                ),
                (r - k.shape[-1])
                // 2 : (
                    -(r - k.shape[-1]) // 2 if -(r - k.shape[-1]) // 2 != 0 else None
                ),
            ] = k
        self.kernel = jax.device_put(self.kernel)

        self.pr, self.pc = self.kernel.shape[-1] // 2, self.kernel.shape[-2] // 2
        XN = jnp.ones((1, 1, m, n))
        self.N = [
            self.boxfilter(XN, self.kernel[i : i + 1, ...])
            for i in range(self.AngleNum)
        ]

    def boxfilter(self, x, k):
        return jax.lax.conv_general_dilated(
            jnp.pad(
                x,
                ((0, 0), (0, 0), (self.pc, self.pc), (self.pr, self.pr)),
                mode="constant",
                constant_values=0,
            ),
            k,
            (1, 1),
            "VALID",
            feature_group_count=x.shape[1],
        )

    def __call__(self, X, y):
        for i in range(self.AngleNum):
            mean_y, mean_x = (
                self.boxfilter(y, self.kernel[i : i + 1, ...]) / self.N[i],
                self.boxfilter(X, self.kernel[i : i + 1, ...]) / self.N[i],
            )
            b = mean_y - mean_x
            X = X + b
        return X


class dual_view_fusion:
    def __init__(self, r, m, n, resampleRatio, eps=1):
        self.r = r
        self.mask = jnp.arange(m)[None, None, :, None]
        self.m, self.n = m, n
        self.eps = eps
        self.resampleRatio = resampleRatio

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = jnp.concatenate([left, middle, right], 2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = jnp.concatenate([left, middle, right], 3)
        return output

    @partial(jit, static_argnums=(0,))
    def boxfilter(self, input):
        return self.diff_y(self.diff_x(input.cumsum(2), self.r).cumsum(3), self.r)

    @partial(jit, static_argnums=(0,))
    def guidedfilter(self, x, y):
        N = self.boxfilter(jnp.ones_like(x))
        mean_x, mean_y = self.boxfilter(x) / N, self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / jnp.clip(var_x, self.eps, None)
        b = mean_y - A * mean_x
        A, b = self.boxfilter(A) / N, self.boxfilter(b) / N
        return A * x + b

    def __call__(self, x, boundary):
        boundary = (
            jax.image.resize(boundary, (1, 1, 1, self.n), method="bilinear")
            / self.resampleRatio
        )
        topSlice, bottomSlice = jnp.split(x, indices_or_sections=2, axis=1)
        mask0, mask1 = self.mask > boundary, self.mask <= boundary
        result0, result1 = self.guidedfilter(bottomSlice, mask0), self.guidedfilter(
            topSlice, mask1
        )
        t = result0 + result1 + 1e-3
        result0, result1 = result0 / t, result1 / t
        return result0 * bottomSlice + result1 * topSlice


class identical_func:
    def __init__(
        self,
    ):
        pass

    def __call__(self, x, boundary):
        return x


class DeStripeModel(hk.Module):
    def __init__(
        self,
        Angle,
        hier_mask,
        hier_ind,
        NI,
        m,
        n,
        resampleRatio,
        KS,
        inc=16,
        GFr=49,
        viewnum=1,
        device="gpu",
    ):
        super().__init__()
        self.NI, self.hier_mask, self.hier_ind, self.inc = NI, hier_mask, hier_ind, inc
        self.ainput = jnp.ones((1, 1))
        self.m, self.n = m, n
        self.Angle = Angle
        gx, gy = self.rotatableKernel(Wsize=1, sigma=1)
        if len(Angle) > 1:
            self.TVfftx, self.inverseTVfftx, self.TVffty, self.inverseTVffty = (
                [],
                [],
                [],
                [],
            )
            for i, A in enumerate(Angle):
                self.fftnt(
                    math.cos(-A / 180 * math.pi) * gx
                    + math.sin(-A / 180 * math.pi) * gy,
                    m,
                    n,
                )
                self.fftn(
                    math.cos(-A / 180 * math.pi + math.pi / 2) * gx
                    + math.sin(-A / 180 * math.pi + math.pi / 2) * gy,
                    m,
                    n,
                )
            self.TVfftx, self.inverseTVfftx = jnp.concatenate(
                self.TVfftx, 0
            ), jnp.concatenate(self.inverseTVfftx, 0)
            self.TVffty, self.inverseTVffty = jnp.concatenate(
                self.TVffty, 0
            ), jnp.concatenate(self.inverseTVffty, 0)
        else:
            if Angle[0] != 0:
                self.TVfftx, self.inverseTVfftx = self.fftnt(
                    math.cos(-Angle[0] / 180 * math.pi) * gx
                    + math.sin(-Angle[0] / 180 * math.pi) * gy,
                    m,
                    n,
                )
                self.TVffty, self.inverseTVffty = self.fftn(
                    math.cos(-Angle[0] / 180 * math.pi + math.pi / 2) * gx
                    + math.sin(-Angle[0] / 180 * math.pi + math.pi / 2) * gy,
                    m,
                    n,
                )
            else:
                self.TVfftx, self.inverseTVfftx = self.fftnt(
                    jnp.array([[1, -1]], dtype=jnp.float32), m, n
                )
                self.TVffty, self.inverseTVffty = self.fftn(
                    jnp.array([[1], [-1]], dtype=jnp.float32), m, n
                )
        self.eigDtD = 1 / (
            jnp.power(jnp.abs(self.TVfftx), 2) + jnp.power(jnp.abs(self.TVffty), 2)
        )
        self.GuidedFilter = GuidedFilterJAX(rx=KS, ry=0, m=m, n=n, Angle=Angle)
        self.p = ResLearning(inc)
        self.edgeProcess = hk.Sequential([CLinear(inc), complexReLUJIT, CLinear(inc)])
        self.latentProcess = hk.Sequential([CLinear(inc), complexReLUJIT, CLinear(inc)])
        self.basep = hk.Sequential(
            [
                hk.Linear(inc),
                jax.nn.relu,
                hk.Linear(inc),
                jax.nn.relu,
                hk.Linear(viewnum),
            ]
        )
        self.merge = hk.Sequential(
            [
                CLinear(inc),
                complexReLUJIT,
                CLinear(inc),
                complexReLUJIT,
                CLinear(viewnum),
            ]
        )
        self.viewnum = viewnum
        self.fuse = (
            dual_view_fusion(GFr, m, n, resampleRatio)
            if viewnum == 2
            else identical_func()
        )

    def rotatableKernel(self, Wsize, sigma):
        k = jnp.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g, gp = jnp.exp(-(k**2) / (2 * sigma**2)), -(k / sigma) * jnp.exp(
            -(k**2) / (2 * sigma**2)
        )
        return g.T * gp, gp.T * g

    def fftnt(self, x, row, col):
        y = jnp.fft.fftshift(jnp.fft.fft2(x, s=(row, col))).reshape(-1)[
            : col * row // 2
        ][None, :, None]
        if len(self.Angle) > 1:
            self.TVfftx.append(y)
            self.inverseTVfftx.append(jnp.conj(y))
        else:
            return y, jnp.conj(y)

    def fftn(self, x, row, col):
        y = jnp.fft.fftshift(jnp.fft.fft2(x, s=(row, col))).reshape(-1)[
            : col * row // 2
        ][None, :, None]
        if len(self.Angle) > 1:
            self.TVffty.append(y)
            self.inverseTVffty.append(jnp.conj(y))
        else:
            return y, jnp.conj(y)

    def fourierResult(self, z, aver):
        return jnp.abs(
            jnp.fft.ifft2(
                jnp.fft.ifftshift(
                    jnp.concatenate(
                        (z, aver * self.basep(self.ainput), jnp.conj(jnp.flip(z, -2))),
                        -2,
                    )
                    .reshape(1, self.m, -1, self.viewnum)
                    .transpose(0, 3, 1, 2)
                )
            )
        )

    def __call__(self, X, Xf, target, boundary):
        aver = X.sum(axis=(2, 3))
        Xf = self.p(Xf)
        w = hk.get_parameter(
            "w",
            self.NI.shape,
            jnp.complex64,
            init=Cmplx_Xavier_Init(self.NI.shape[0], self.NI.shape[1]),
        )
        Xfcx = jnp.sum(jnp.einsum("knc,kn->knc", Xf[self.NI, :], w), 0)
        Xf_tvx = jnp.concatenate((Xfcx, Xf[self.hier_mask, :]), 0)[
            self.hier_ind, :
        ].reshape(len(self.Angle), -1, self.inc)
        Xf_tvx, Xf_tvy = Xf_tvx * self.TVfftx, Xf * self.TVffty
        X_fourier = []
        for x, y, inverseTVfftx, inverseTVffty, eigDtD in zip(
            Xf_tvx, Xf_tvy, self.inverseTVfftx, self.inverseTVffty, self.eigDtD
        ):
            X_fourier.append(
                complexReLUJIT(
                    self.latentProcess(
                        complexReLUJIT(self.edgeProcess(x) * inverseTVfftx)
                        + complexReLUJIT(self.edgeProcess(y) * inverseTVffty)
                    )
                    * eigDtD
                )
            )
        X_fourier = self.merge(jnp.concatenate(X_fourier, -1))
        outputGNNraw = self.fourierResult(X_fourier, aver)
        outputGNN = self.fuse(outputGNNraw, boundary)
        outputLR = self.GuidedFilter(target, outputGNN)
        return outputGNNraw, outputGNN, outputLR


class GuidedFilterLoss:
    def __init__(self, r, eps=1e-9):
        self.r, self.eps = r, eps

    def diff_x(self, input, r):
        return input[:, :, 2 * r + 1 :, :] - input[:, :, : -2 * r - 1, :]

    def diff_y(self, input, r):
        return input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]

    @partial(jit, static_argnums=(0,))
    def boxfilter(self, input):
        return self.diff_x(
            self.diff_y(
                jnp.pad(
                    input,
                    ((0, 0), (0, 0), (self.r + 1, self.r), (self.r + 1, self.r)),
                    mode="constant",
                ).cumsum(3),
                self.r,
            ).cumsum(2),
            self.r,
        )

    @partial(jit, static_argnums=(0,))
    def __call__(self, x, y):
        N = self.boxfilter(jnp.ones_like(x))
        mean_x, mean_y = self.boxfilter(x) / N, self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / jnp.clip(var_x, self.eps, None)
        b = mean_y - A * mean_x
        A, b = self.boxfilter(A) / N, self.boxfilter(b) / N
        return A * x + b


class Loss:
    def __init__(self, train_params, shape_params, device="gpu"):
        super().__init__()
        self.lambda_tv = train_params["lambda_tv"]
        self.lambda_hessian = train_params["lambda_hessian"]
        self.angleOffset = shape_params["angle_offset"]
        self.sampling = train_params["sampling_in_MSEloss"]
        self.f = train_params["isotropic_hessian"]
        if train_params["hessian_kernel_sigma"] > 0.5:
            self.DGaussxx, self.DGaussyy, self.DGaussxy = self.generateHessianKernel(
                train_params["hessian_kernel_sigma"]
            )
        else:
            self.DGaussxx, self.DGaussyy, self.DGaussxy = self.generateHessianKernel2(
                train_params["hessian_kernel_sigma"], shape_params
            )

        # self.Dy, self.Dx = (
        #     jnp.array([[1], [-1]], dtype=jnp.float32)[None, None],
        #     jnp.array([[1, -1]], dtype=jnp.float32)[None, None],
        # )
        offset = sum(shape_params["angle_offset"]) / len(shape_params["angle_offset"])
        gx, gy = self.rotatableKernel(Wsize=3, sigma=1)
        self.Dx = (
            math.cos(-offset / 180 * math.pi) * gx
            + math.sin(-offset / 180 * math.pi) * gy
        )
        self.Dy = (
            math.cos(-offset / 180 * math.pi + math.pi / 2) * gx
            + math.sin(-offset / 180 * math.pi + math.pi / 2) * gy
        )
        self.Dx, self.Dy = self.Dx[None, None], self.Dy[None, None]

        self.GuidedFilterLoss = GuidedFilterLoss(
            r=train_params["GF_kernel_size_train"],
            eps=train_params["loss_eps"],
        )

    def generateHessianKernel2(self, Sigma, shape_params):
        Wsize = math.ceil(3 * Sigma)
        KernelSize = 2 * (2 * Wsize + 1) - 1
        gx, gy = self.rotatableKernel(Wsize, Sigma)
        md = shape_params["md"] if shape_params["is_vertical"] else shape_params["nd"]
        nd = shape_params["nd"] if shape_params["is_vertical"] else shape_params["md"]
        gxFFT2, gyFFT2 = jnp.fft.fft2(gx, s=(md, nd)), jnp.fft.fft2(gy, s=(md, nd))
        DGaussxx = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        DGaussxy = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        DGaussyy = torch.zeros(len(self.angleOffset), 1, KernelSize, KernelSize)
        for i, A in enumerate(self.angleOffset):
            a, b, c, d = (
                math.cos(-A / 180 * math.pi),
                math.sin(-A / 180 * math.pi),
                math.cos(-A / 180 * math.pi + math.pi / 2),
                math.sin(-A / 180 * math.pi + math.pi / 2),
            )
            DGaussxx[i] = (
                fft.ifft2((a * gxFFT2 + b * gyFFT2) * (a * gxFFT2 + b * gyFFT2))
                .real[None, None, :KernelSize, :KernelSize]
                .float()
            )
            DGaussyy[i] = (
                fft.ifft2((c * gxFFT2 + d * gyFFT2) * (c * gxFFT2 + d * gyFFT2))
                .real[None, None, :KernelSize, :KernelSize]
                .float()
            )
            DGaussxy[i] = (
                fft.ifft2((c * gxFFT2 + d * gyFFT2) * (a * gxFFT2 + b * gyFFT2))
                .real[None, None, :KernelSize, :KernelSize]
                .float()
            )
        return (
            jnp.asarray(DGaussxx.data.numpy()),
            jnp.asarray(DGaussyy.data.numpy()),
            jnp.asarray(DGaussxy.data.numpy()),
        )

    def rotatableKernel(self, Wsize, sigma):
        k = jnp.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g = jnp.exp(-(k**2) / (2 * sigma**2))
        gp = -(k / sigma) * jnp.exp(-(k**2) / (2 * sigma**2))
        return g.T * gp, gp.T * g

    def generateHessianKernel(self, Sigma):
        tmp = np.linspace(
            -1 * np.ceil(Sigma * 6), np.ceil(Sigma * 6), int(np.ceil(Sigma * 6) * 2 + 1)
        )
        X, Y = np.meshgrid(tmp, tmp, indexing="ij")
        DGaussxx = torch.from_numpy(
            1
            / (2 * math.pi * Sigma**4)
            * (X**2 / Sigma**2 - 1)
            * np.exp(-(X**2 + Y**2) / (2 * Sigma**2))
        )[None, None, :, :]
        DGaussxy = torch.from_numpy(
            1
            / (2 * math.pi * Sigma**6)
            * (X * Y)
            * np.exp(-(X**2 + Y**2) / (2 * Sigma**2))
        )[None, None, :, :]
        DGaussyy = DGaussxx.transpose(3, 2)
        DGaussxx, DGaussxy, DGaussyy = (
            DGaussxx.float().data.numpy(),
            DGaussxy.float().data.numpy(),
            DGaussyy.float().data.numpy(),
        )
        Gaussxx, Gaussxy, Gaussyy = [], [], []
        for A in self.angleOffset:
            Gaussxx.append(
                scipy.ndimage.rotate(DGaussxx, A, axes=(-2, -1), reshape=False)
            )
            Gaussyy.append(
                scipy.ndimage.rotate(DGaussyy, A, axes=(-2, -1), reshape=False)
            )
            Gaussxy.append(
                scipy.ndimage.rotate(DGaussxy, A, axes=(-2, -1), reshape=False)
            )
        Gaussxx, Gaussyy, Gaussxy = (
            np.concatenate(Gaussxx, 0),
            np.concatenate(Gaussyy, 0),
            np.concatenate(Gaussxy, 0),
        )
        return jnp.asarray(Gaussyy), jnp.asarray(Gaussxx), jnp.asarray(Gaussxy)

    @partial(jit, static_argnums=0)
    def TotalVariation(self, x, target, Dx, Dy):
        return (
            jnp.abs(
                jax.lax.conv_general_dilated(
                    x, Dx, (1, 1), "VALID", feature_group_count=x.shape[1]
                )
            ).sum()
            + jnp.abs(
                jax.lax.conv_general_dilated(
                    x - target, Dy, (1, 1), "VALID", feature_group_count=x.shape[1]
                )
            ).sum()
        )

    @partial(jit, static_argnums=0)
    def HessianRegularizationLoss(self, x, target, DGaussxx, DGaussyy, DGaussxy):
        return (
            jnp.abs(
                jax.lax.conv_general_dilated(
                    x, DGaussxx, (1, 1), "VALID", feature_group_count=x.shape[1]
                )
            ).sum()
            + jnp.abs(
                jax.lax.conv_general_dilated(
                    x - target,
                    DGaussyy,
                    (1, 1),
                    "VALID",
                    feature_group_count=x.shape[1],
                )
            ).sum()
            + (2 if self.f else 1)
            * jnp.abs(
                jax.lax.conv_general_dilated(
                    x - target,
                    DGaussxy,
                    (1, 1),
                    "VALID",
                    feature_group_count=x.shape[1],
                )
            ).sum()
        )

    def __call__(
        self,
        params,
        network,
        inputs,
        targets,
        smoothedTarget,
        map,
    ):
        outputGNNraw, outputGNN, outputLR = network.apply(params, **inputs)
        mse = jnp.sum(
            jnp.abs(smoothedTarget - self.GuidedFilterLoss(outputGNNraw, outputGNNraw))
        ) + jnp.sum(
            (jnp.abs(targets - outputGNN) * map)[
                :, :, :: self.sampling, :: self.sampling
            ]
        )
        tv = 1 * self.TotalVariation(
            outputGNN, targets, self.Dx, self.Dy
        ) + 1 * self.TotalVariation(outputLR, targets, self.Dx, self.Dy)
        hessian = 1 * self.HessianRegularizationLoss(
            outputLR, targets, self.DGaussxx, self.DGaussyy, self.DGaussxy
        ) + 1 * self.HessianRegularizationLoss(
            outputGNN, targets, self.DGaussxx, self.DGaussyy, self.DGaussxy
        )
        return 1 * mse + self.lambda_tv * tv + self.lambda_hessian * hessian, [
            outputGNNraw,
            outputGNN,
            outputLR,
        ]
