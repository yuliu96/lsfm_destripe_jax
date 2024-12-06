import jaxwt
import jax.numpy as jnp


def wave_rec(recon, hX, x_base, kernel, only_vertical=True):
    y_dict = jaxwt.wavedec2(recon[:, :, :-1, :-1], kernel, level=6, mode="reflect")
    X_dict = jaxwt.wavedec2(hX[:, :, :-1, :-1], kernel, level=6, mode="reflect")
    x_base_dict = jaxwt.wavedec2(
        x_base[:, :, :-1, :-1], kernel, level=6, mode="reflect"
    )
    if only_vertical:
        y_dict_new = [x_base_dict[0]]
    else:
        y_dict_new = [x_base_dict[0]]
    for detail, target in zip(y_dict[1:], X_dict[1:]):
        if only_vertical:
            y_dict_new.append(
                [
                    jnp.where(
                        jnp.abs(detail[0]) > jnp.abs(target[0]), detail[0], target[0]
                    ),
                    jnp.where(
                        jnp.abs(detail[1]) > jnp.abs(target[1]), detail[1], target[1]
                    ),
                    jnp.where(
                        jnp.abs(detail[2]) > jnp.abs(target[2]), detail[2], target[2]
                    ),
                ]
            )
        else:
            y_dict_new.append(
                [
                    jnp.where(
                        jnp.abs(detail[0]) < jnp.abs(target[0]), detail[0], target[0]
                    ),
                    jnp.where(
                        jnp.abs(detail[1]) < jnp.abs(target[1]), detail[1], target[1]
                    ),
                    jnp.where(
                        jnp.abs(detail[2]) < jnp.abs(target[2]), detail[2], target[2]
                    ),
                ]
            )
    recon = jnp.pad(
        jaxwt.waverec2(y_dict_new, kernel), ((0, 0), (0, 0), (0, 1), (0, 1)), "reflect"
    )
    return torch.from_numpy(np.asarray(recon)).cuda()


class GuidedFilterHR_fast(nn.Module):
    def __init__(self, rX, rY, angleList, eps=1e-6, device="cpu"):
        super(GuidedFilterHR_fast, self).__init__()
        self.rX = rX
        self.rY = rY
        self.eps = eps
        self.boxfilter_detail = [
            BoxFilter(rX[0], rY[0], Angle=0, device=device) for Angle in angleList
        ]
        self.boxfilter_base = [
            BoxFilter(rX[1], rY[1], Angle=0, device=device) for Angle in angleList
        ]
        self.N = None
        self.angleList = angleList
        self.crop = None
        self.GaussianKernel = torch.tensor(
            np.ones((rX[1], 1)) * np.ones((1, rX[1])), dtype=torch.float32
        )
        self.GaussianKernel = (self.GaussianKernel / self.GaussianKernel.sum())[
            None, None
        ].to(device)
        self.GaussianKernelpadding = (
            self.GaussianKernel.shape[-2] // 2,
            self.GaussianKernel.shape[-1] // 2,
        )
        self.device = device

    def forward(self, xx, yy, coor, hX):
        # yy = F.interpolate(yy, size = hX.shape[-2:], mode = "bilinear")
        # xx = F.interpolate(xx, size = hX.shape[-2:], mode = "bilinear")
        # return self.GF(hX, yy, hX, stripe_lr)
        hXX = copy.deepcopy(hX)

        # recon = hX - F.interpolate(xx-yy, size = hX.shape[-2:], mode = "bilinear")
        yy = jnp.array(yy.cpu().data.numpy())
        hX = jax.scipy.ndimage.map_coordinates(
            np.array(xx.cpu().data.numpy()), coor, order=1, mode="reflect"
        )[None, None]
        recon = jax.scipy.ndimage.map_coordinates(yy, coor, order=1, mode="reflect")[
            None, None
        ]
        recon = wave_rec(recon, hXX, recon, "db2", False)
        hX = wave_rec(hX, hXX, hX, "db2", False)
        # recon = torch.from_numpy(np.asarray(recon))
        recon, hX = self.GF(hX, recon.cuda(), hXX)
        return recon

    def GF(self, xx, yy, hX):
        hX_original = copy.deepcopy(hX)
        with torch.no_grad():
            if self.crop is None:
                self.crop = torchvision.transforms.CenterCrop(xx.size()[-2:])
            for i, Angle in enumerate((-1 * np.array(self.angleList)).tolist()):
                hX_original = torch.from_numpy(
                    scipy.ndimage.rotate(
                        hX_original.cpu(),
                        Angle,
                        axes=(-2, -1),
                        reshape=True,
                        order=1,
                        mode="reflect",
                    )
                ).cuda()
                x_1 = torch.from_numpy(
                    scipy.ndimage.rotate(
                        xx.cpu(),
                        Angle,
                        axes=(-2, -1),
                        reshape=True,
                        order=1,
                        mode="reflect",
                    )
                ).cuda()
                y_1 = torch.from_numpy(
                    scipy.ndimage.rotate(
                        yy.cpu(),
                        Angle,
                        axes=(-2, -1),
                        reshape=True,
                        order=1,
                        mode="reflect",
                    )
                ).cuda()
                hx_1 = torch.from_numpy(
                    scipy.ndimage.rotate(
                        hX.cpu(),
                        Angle,
                        axes=(-2, -1),
                        reshape=True,
                        order=1,
                        mode="reflect",
                    )
                ).cuda()
                pad = np.zeros(4, dtype=np.int32)
                if x_1.shape[-1] % 2 == 0:
                    pad[1] = 1
                else:
                    pass
                if x_1.shape[-2] % 2 == 0:
                    pad[3] = 1
                else:
                    pass
                pad = tuple(pad)
                x_1 = F.pad(x_1, pad, "reflect")
                y_1 = F.pad(y_1, pad, "reflect")
                hx_1 = F.pad(hx_1, pad, "reflect")
                hX_original = F.pad(hX_original, pad, "reflect")
                h_x, w_x = x_1.size()[-2:]
                # x_1 = wave_rec(y_1, x_1, y_1, "db8")
                # hx_1 = wave_rec(y_1, hx_1,  y_1, "db8")
                b = y_1 - x_1

                b = (
                    F.pad(b, (0, 0, 89 // 2, 89 // 2), "reflect")
                    .unfold(2, 89, 1)
                    .median(-1)[0]
                )
                # b = F.pad(b, (5, 5, 0, 0), "reflect").unfold(3, 11, 1).median(-1)[0]
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
                xx = self.crop(
                    torchvision.transforms.functional.rotate(
                        x_1,
                        -Angle,
                        expand=True,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    )
                )
                yy = self.crop(
                    torchvision.transforms.functional.rotate(
                        y_1,
                        -Angle,
                        expand=True,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    )
                )
                hX = self.crop(
                    torchvision.transforms.functional.rotate(
                        hx_1,
                        -Angle,
                        expand=True,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    )
                )
                hX_original = self.crop(
                    torchvision.transforms.functional.rotate(
                        hX_original,
                        -Angle,
                        expand=True,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    )
                )
        return hX, hX_original
