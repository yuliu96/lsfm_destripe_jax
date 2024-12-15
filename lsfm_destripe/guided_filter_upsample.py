import jaxwt
import jax.numpy as jnp
import copy
import jax
import numpy as np
import scipy
from lsfm_destripe.utils import crop_center
from lsfm_destripe.constant import WaveletDetailTuple2d


def wave_rec(
    recon,
    hX,
    x_base,
    kernel,
    only_vertical=True,
):
    y_dict = jaxwt.wavedec2(
        recon[:, :, :-1, :-1],
        kernel,
        level=6,
        mode="reflect",
    )
    X_dict = jaxwt.wavedec2(
        hX[:, :, :-1, :-1],
        kernel,
        level=6,
        mode="reflect",
    )
    x_base_dict = jaxwt.wavedec2(
        x_base[:, :, :-1, :-1],
        kernel,
        level=6,
        mode="reflect",
    )
    y_dict_new = [x_base_dict[0]]
    for detail, target in zip(y_dict[1:], X_dict[1:]):
        if only_vertical:
            y_dict_new.append(
                [
                    jnp.where(
                        jnp.abs(detail[0]) > jnp.abs(target[0]),
                        detail[0],
                        target[0],
                    ),
                    jnp.where(
                        jnp.abs(detail[1]) > jnp.abs(target[1]),
                        detail[1],
                        target[1],
                    ),
                    jnp.where(
                        jnp.abs(detail[2]) > jnp.abs(target[2]),
                        detail[2],
                        target[2],
                    ),
                ]
            )
        else:
            y_dict_new.append(
                [
                    jnp.where(
                        jnp.abs(detail[0]) < jnp.abs(target[0]),
                        detail[0],
                        target[0],
                    ),
                    jnp.where(
                        jnp.abs(detail[1]) < jnp.abs(target[1]),
                        detail[1],
                        target[1],
                    ),
                    jnp.where(
                        jnp.abs(detail[2]) < jnp.abs(target[2]),
                        detail[2],
                        target[2],
                    ),
                ]
            )
    recon = jnp.pad(
        jaxwt.waverec2(y_dict_new, kernel),
        ((0, 0), (0, 0), (0, 1), (0, 1)),
        "reflect",
    )
    return recon


class GuidedFilterHR_fast:
    def __init__(
        self,
        rx,
        ry,
    ):
        self.rx = rx
        self.ry = ry

    def __call__(
        self,
        xx,
        yy,
        hX,
        fusion_mask,
        angle_offset_individual,
        fidelity_first,
        backend,
    ):
        m, n = hX.shape[-2:]
        hXX = copy.deepcopy(hX)
        hX = jax.image.resize(
            xx,
            (
                1,
                xx.shape[1],
                m,
                n,
            ),
            method="lanczos5",
        )
        recon = jax.image.resize(
            yy,
            (
                1,
                1,
                m,
                n,
            ),
            method="lanczos5",
        )

        y = jnp.ones_like(fusion_mask)
        for i, angle_list in enumerate(angle_offset_individual):
            hX_slice = hX[:, i : i + 1, :, :]
            hXX_slice = hXX[:, i : i + 1, :, :]

            recon_slice = wave_rec(
                recon,
                hXX_slice,
                recon,
                "db2",
                False,
            )
            hX_slice = wave_rec(
                hX_slice,
                hXX_slice,
                hX_slice,
                "db2",
                False,
            )

            y = y.at[:, i : i + 1, :, :].set(
                self.GF(
                    hX_slice,
                    recon_slice,
                    hXX_slice,
                    angle_list,
                    fidelity_first,
                )
            )
        y = (10**y) * fusion_mask
        return y.sum(1, keepdims=True)

    def GF(
        self,
        xx,
        yy,
        hX,
        angle_list,
        fidelity_first,
    ):
        _, _, m, n = hX.shape
        for i, Angle in enumerate((-1 * np.array(angle_list)).tolist()):
            x_1 = jnp.asarray(
                scipy.ndimage.rotate(
                    xx,
                    Angle,
                    axes=(-2, -1),
                    reshape=True,
                    order=1,
                    mode="reflect",
                )
            )
            y_1 = jnp.asarray(
                scipy.ndimage.rotate(
                    yy,
                    Angle,
                    axes=(-2, -1),
                    reshape=True,
                    order=1,
                    mode="reflect",
                )
            )
            hx_1 = jnp.asarray(
                scipy.ndimage.rotate(
                    hX,
                    Angle,
                    axes=(-2, -1),
                    reshape=True,
                    order=1,
                    mode="reflect",
                )
            )
            pad = [(0, 0), (0, 0)]
            if x_1.shape[-2] % 2 == 0:
                pad = pad + [(0, 1)]
            else:
                pad = pad + [(0, 0)]
            if x_1.shape[-1] % 2 == 0:
                pad = pad + [(0, 1)]
            else:
                pad = pad + [(0, 0)]
            pad = tuple(pad)
            x_1 = jnp.pad(x_1, pad, "reflect")
            y_1 = jnp.pad(y_1, pad, "reflect")
            hx_1 = jnp.pad(hx_1, pad, "reflect")
            b = y_1 - x_1
            b = jnp.median(
                jax.lax.conv_general_dilated_patches(
                    jnp.pad(
                        b,
                        (
                            (0, 0),
                            (0, 0),
                            (49 // 2, 49 // 2),
                            (0, 0),
                        ),
                        "reflect",
                    ),
                    [49, 1],
                    [1, 1],
                    "VALID",
                ),
                1,
                keepdims=True,
            )
            for e in range(3):
                k = self.rx // 3 // 2 * 2 + 1
                b = jnp.median(
                    jax.lax.conv_general_dilated_patches(
                        jnp.pad(
                            b,
                            (
                                (0, 0),
                                (0, 0),
                                (k // 2, k // 2),
                                (0, 0) if e < 2 else (self.ry // 2, self.ry // 2),
                            ),
                            "reflect",
                        ),
                        [k, 1 if e < 2 else self.ry],
                        [1, 1],
                        "VALID",
                    ),
                    1,
                    keepdims=True,
                )
            x_1 = wave_rec(
                x_1 if fidelity_first else y_1,
                x_1 + b,
                x_1 + b if fidelity_first else y_1,
                "db12",
                False if fidelity_first else True,
            )
            hx_1 = wave_rec(
                hx_1 if fidelity_first else y_1,
                hx_1 + b,
                hx_1 + b if fidelity_first else y_1,
                "db12",
                False if fidelity_first else True,
            )
            xx = crop_center(
                scipy.ndimage.rotate(
                    x_1,
                    -Angle,
                    axes=(-2, -1),
                    reshape=True,
                    order=1,
                    mode="reflect",
                ),
                m,
                n,
            )
            yy = crop_center(
                scipy.ndimage.rotate(
                    y_1,
                    -Angle,
                    axes=(-2, -1),
                    reshape=True,
                    order=1,
                    mode="reflect",
                ),
                m,
                n,
            )
            hX = crop_center(
                scipy.ndimage.rotate(
                    hx_1,
                    -Angle,
                    axes=(-2, -1),
                    reshape=True,
                    order=1,
                    mode="reflect",
                ),
                m,
                n,
            )

        return hX
