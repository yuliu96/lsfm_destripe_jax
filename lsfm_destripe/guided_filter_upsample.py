import jaxwt
import jax.numpy as jnp
import copy
import jax
import numpy as np
import scipy
from utils import crop_center


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
        angleList,
    ):
        self.rx = rx
        self.ry = ry
        self.angleList = angleList

    def __call__(self, xx, yy, coor, hX):
        hXX = copy.deepcopy(hX)
        hX = jax.scipy.ndimage.map_coordinates(
            xx,
            coor,
            order=1,
            mode="reflect",
        )[None, None]
        recon = jax.scipy.ndimage.map_coordinates(
            yy,
            coor,
            order=1,
            mode="reflect",
        )[None, None]
        recon = wave_rec(
            recon,
            hXX,
            recon,
            "db2",
            False,
        )
        hX = wave_rec(
            hX,
            hXX,
            hX,
            "db2",
            False,
        )
        recon = self.GF(
            hX,
            recon,
            hXX,
        )
        return recon

    def GF(self, xx, yy, hX):
        _, _, m, n = hX.shape
        for i, Angle in enumerate((-1 * np.array(self.angleList)).tolist()):
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
                    hX.cpu(),
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
                            (self.rx // 2, self.rx // 2),
                            (self.ry // 2, self.ry // 2),
                        ),
                        "reflect",
                    ),
                    [self.rx, self.ry],
                    [1, 1],
                    "VALID",
                ),
                1,
            )
            x_1 = wave_rec(
                y_1,
                x_1 + b,
                y_1,
                "db8",
            )
            hx_1 = wave_rec(
                y_1,
                hx_1 + b,
                y_1,
                "db8",
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
