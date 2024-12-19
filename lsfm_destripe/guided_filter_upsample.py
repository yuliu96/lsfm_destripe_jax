import jaxwt
import jax.numpy as jnp
import copy
import jax
import numpy as np
from lsfm_destripe.constant import WaveletDetailTuple2d
import torch
import ptwt
import pywt
import torch.nn.functional as F


def wave_rec(
    recon,
    hX,
    kernel,
    mode,
):

    y_dict = ptwt.wavedec2(
        recon[:, :, :-1, :-1], pywt.Wavelet(kernel), level=6, mode="constant"
    )
    X_dict = ptwt.wavedec2(
        hX[:, :, :-1, :-1], pywt.Wavelet(kernel), level=6, mode="constant"
    )
    x_base_dict = [y_dict[0]]

    mask_dict = []
    for l, (detail, target) in enumerate(zip(y_dict[1:], X_dict[1:])):
        mask_dict.append(
            [
                torch.abs(detail[0]) < torch.abs(target[0]),
                torch.abs(detail[1]) < torch.abs(target[1]),
                torch.abs(detail[2]) < torch.abs(target[2]),
            ]
        )

    for l, (detail, target, mask) in enumerate(zip(y_dict[1:], X_dict[1:], mask_dict)):
        if mode == 1:
            x_base_dict.append(
                WaveletDetailTuple2d(
                    torch.where(
                        ~mask[0],
                        detail[0],
                        target[0],
                    ),
                    torch.where(
                        mask[1],
                        detail[1],
                        target[1],
                    ),
                    torch.where(
                        ~mask[2],
                        detail[2],
                        target[2],
                    ),
                )
            )  # torch.sign(detail[1])*target[1].abs()
        else:
            x_base_dict.append(
                WaveletDetailTuple2d(
                    torch.where(
                        ~mask[0],
                        detail[0],
                        torch.sign(detail[0]) * target[0].abs(),
                    ),
                    torch.where(
                        mask[1],
                        detail[1],
                        torch.sign(detail[1]) * target[1].abs(),
                    ),
                    torch.where(
                        ~mask[2],
                        detail[2],
                        torch.sign(detail[2]) * target[2].abs(),
                    ),
                )
            )  # torch.sign(detail[1])*target[1].abs()
    x_base_dict = tuple(x_base_dict)
    recon = ptwt.waverec2(x_base_dict, pywt.Wavelet(kernel))
    recon = F.pad(recon, (0, 1, 0, 1), "reflect")
    return recon


class GuidedUpsample:
    def __init__(
        self,
        rx,
        ry,
        mode,
        device,
    ):
        self.rx = rx
        self.ry = ry // 2
        self.device = device
        self.mode = mode

    def __call__(
        self,
        xx,
        yy,
        hX,
        targetd,
        target,
        coor,
        fusion_mask,
        angle_offset_individual,
        backend,
    ):
        if backend == "jax":
            recon = (
                jax.scipy.ndimage.map_coordinates(
                    yy - targetd, coor, order=1, mode="reflect"
                )[None, None]
                + target
            )
            recon = torch.from_numpy(np.array(recon, copy=True)).to(self.device)
            hX = torch.from_numpy(np.array(hX)).to(self.device)
            fusion_mask = np.asarray(fusion_mask)
        m, n = hX.shape[-2:]

        y = np.ones_like(fusion_mask)

        for i, angle_list in enumerate(angle_offset_individual):
            hX_slice = hX[:, i : i + 1, :, :]

            y[:, i : i + 1, :, :] = (
                self.GF(
                    recon,
                    hX_slice,
                    angle_list,
                )
                .cpu()
                .data.numpy()
            )
        y = (10**y) * fusion_mask
        return y.sum(1, keepdims=True)

    def GF(
        self,
        yy,
        hX,
        angle_list,
    ):
        hX_original = copy.deepcopy(hX)
        _, _, m, n = hX.shape
        for i, Angle in enumerate((-1 * np.array(angle_list)).tolist()):
            b = yy - hX
            rx = self.rx  # // 3 // 2 * 2 + 1
            l = np.arange(rx) - rx // 2
            l = np.round(l * np.tan(np.deg2rad(-Angle))).astype(np.int32)
            for _ in range(1):
                b_batch = torch.zeros(rx, 1, 1, m, n)
                for ind, r in enumerate(range(rx)):
                    data = F.pad(b, (l.max(), l.max(), rx // 2, rx // 2), "reflect")
                    b_batch[ind] = data[
                        :, :, r : r + m, l[ind] - l.min() : l[ind] - l.min() + n
                    ].cpu()
                b = torch.median(b_batch, 0)[0]
                if _ in range(self.ry):
                    b = torch.median(
                        torch.stack(
                            (
                                b,
                                torch.roll(b, 1, 2),
                                torch.roll(b, -1, 2),
                                torch.roll(b, 1, 3),
                                torch.roll(b, -1, 3),
                            ),
                            0,
                        ),
                        0,
                    )[0]
            b = b.to(self.device)
            hX = hX + b

        hX = wave_rec(hX, hX_original, "db2", mode=self.mode)

        return hX
