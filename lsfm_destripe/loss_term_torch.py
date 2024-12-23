import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import pywt
import ptwt


class Loss_torch(nn.Module):
    def __init__(
        self,
        train_params,
        shape_params,
    ):
        super(Loss_torch, self).__init__()
        self.lambda_tv = train_params["lambda_tv"]
        self.lambda_hessian = train_params["lambda_hessian"]
        self.lambda_masking_mse = train_params["lambda_masking_mse"]
        self.angleOffset = shape_params["angle_offset"]

        DGaussxx_f, DGaussyy_f, DGaussxy_f = [], [], []
        for i in range(len(self.angleOffset)):
            xx, yy, xy = self.generateHessianKernel(
                train_params["hessian_kernel_sigma"],
                shape_params,
                [self.angleOffset[i] + 0.0],
            )
            DGaussxx_f.append(xx)
            DGaussyy_f.append(yy)
            DGaussxy_f.append(xy)
        DGaussxx_f = torch.cat(DGaussxx_f, 0)
        DGaussyy_f = torch.cat(DGaussyy_f, 0)
        DGaussxy_f = torch.cat(DGaussxy_f, 0)

        DGaussxx, DGaussyy, DGaussxy = [], [], []
        for i in range(len(self.angleOffset)):
            xx, yy, xy = self.generateHessianKernel(
                train_params["hessian_kernel_sigma"],
                shape_params,
                np.rad2deg(
                    np.arctan(
                        shape_params["r"]
                        * np.tan(
                            np.deg2rad(
                                self.angleOffset[i] + np.array([-2.0, -1, 0, 1, 2])
                            )
                        )
                    )
                ).tolist(),
            )
            DGaussxx.append(xx)
            DGaussyy.append(yy)
            DGaussxy.append(xy)

        DGaussxx = torch.cat(DGaussxx, 0)
        DGaussyy = torch.cat(DGaussyy, 0)
        DGaussxy = torch.cat(DGaussxy, 0)

        Dx_f, Dy_f = [], []
        Dx, Dy = [], []
        for i in range(len(self.angleOffset)):
            Dx_fs, Dy_fs = self.total_variation_kernel([self.angleOffset[i] + 0.0])
            Dxs, Dys = self.total_variation_kernel(
                [
                    np.rad2deg(
                        np.arctan(
                            shape_params["r"]
                            * np.tan(np.deg2rad(self.angleOffset[i] + 0.0))
                        )
                    )
                ]
            )
            Dx_f.append(Dx_fs)
            Dy_f.append(Dy_fs)
            Dx.append(Dxs)
            Dy.append(Dys)

        Dx_f = torch.cat(Dx_f, 0)
        Dy_f = torch.cat(Dy_f, 0)
        Dx = torch.cat(Dx, 0)
        Dy = torch.cat(Dy, 0)

        DGaussxy = (
            DGaussxy
            / torch.abs(DGaussxy).sum((-2, -1), keepdim=True)
            * torch.abs(Dx.repeat_interleave(5, 0)).sum((-2, -1), keepdim=True)
        )
        DGaussyy = (
            DGaussyy
            / torch.abs(DGaussyy).sum((-2, -1), keepdim=True)
            * torch.abs(Dx.repeat_interleave(5, 0)).sum((-2, -1), keepdim=True)
        )
        DGaussxx = (
            DGaussxx
            / torch.abs(DGaussxx).sum((-2, -1), keepdim=True)
            * torch.abs(Dx.repeat_interleave(5, 0)).sum((-2, -1), keepdim=True)
        )

        DGaussxy_f = (
            DGaussxy_f
            / torch.abs(DGaussxy_f).sum((-2, -1), keepdim=True)
            * torch.abs(Dx_f).sum((-2, -1), keepdim=True)
        )
        DGaussyy_f = (
            DGaussyy_f
            / torch.abs(DGaussyy_f).sum((-2, -1), keepdim=True)
            * torch.abs(Dx_f).sum((-2, -1), keepdim=True)
        )
        DGaussxx_f = (
            DGaussxx_f
            / torch.abs(DGaussxx_f).sum((-2, -1), keepdim=True)
            * torch.abs(Dx_f).sum((-2, -1), keepdim=True)
        )

        self.r = int(shape_params["r"])

        self.p_hessian = (
            DGaussxx.shape[3] // 2,
            DGaussxx.shape[3] // 2,
            DGaussxx.shape[2] // 2,
            DGaussxx.shape[2] // 2,
        )
        self.p_tv = (
            Dx.shape[3] // 2,
            Dx.shape[3] // 2,
            Dx.shape[2] // 2,
            Dx.shape[2] // 2,
        )

        p_hessian = (
            int((DGaussxx.shape[2] + (DGaussxx.shape[2] - 1) * (self.r - 1) + 1) / 2)
            - 1
        )
        p_tv = int((Dx.shape[2] + (Dx.shape[2] - 1) * (self.r - 1) + 1) / 2) - 1

        self.p_hessian_f = (p_hessian, p_hessian, self.p_hessian[2], self.p_hessian[3])
        self.p_tv_f = (p_tv, p_tv, self.p_tv[2], self.p_tv[3])

        self.GF_pad = train_params["max_pool_kernel_size"]

        self.total_variation_cal = (
            self.TotalVariationLoss
            if len(self.angleOffset) > 1
            else self.TotalVariationLoss_plain
        )

        self.hessian_cal = (
            self.HessianRegularizationLoss
            if len(self.angleOffset) > 1
            else self.HessianRegularizationLoss_plain
        )

        self.register_buffer("Dx_f", Dx_f.to(torch.float))
        self.register_buffer("Dy_f", Dy_f.to(torch.float))
        self.register_buffer("DGaussxx_f", DGaussxx_f.to(torch.float))
        self.register_buffer("DGaussyy_f", DGaussyy_f.to(torch.float))
        self.register_buffer("DGaussxy_f", DGaussxy_f.to(torch.float))

        self.register_buffer("Dx", Dx.to(torch.float))
        self.register_buffer("Dy", Dy.to(torch.float))
        self.register_buffer("DGaussxx", DGaussxx.to(torch.float))
        self.register_buffer("DGaussyy", DGaussyy.to(torch.float))
        self.register_buffer("DGaussxy", DGaussxy.to(torch.float))

    def total_variation_kernel(
        self,
        angle_list,
    ):
        gx, gy = self.rotatableKernel(Wsize=3, sigma=1)
        Dx_ = []
        Dy_ = []
        for i in angle_list:
            Dx = math.cos(-i / 180 * math.pi) * gx + math.sin(-i / 180 * math.pi) * gy
            Dy = (
                math.cos(-i / 180 * math.pi + math.pi / 2) * gx
                + math.sin(-i / 180 * math.pi + math.pi / 2) * gy
            )
            Dx_.append(Dx)
            Dy_.append(Dy)
        Dx_ = np.stack(Dx_, 0)[:, None]
        Dy_ = np.stack(Dy_, 0)[:, None]
        Dx_ = Dx_ - Dx_.mean(axis=(-2, -1), keepdims=True)
        Dy_ = Dy_ - Dy_.mean(axis=(-2, -1), keepdims=True)
        return torch.from_numpy(Dx_), torch.from_numpy(Dy_)

    def generateHessianKernel(
        self,
        Sigma,
        shape_params,
        angleOffset,
    ):
        Wsize = math.ceil(3 * Sigma)
        KernelSize = 2 * (2 * Wsize + 1) - 1
        gx, gy = self.rotatableKernel(Wsize, Sigma)

        md = shape_params["md"] if shape_params["is_vertical"] else shape_params["nd"]
        nd = shape_params["nd"] if shape_params["is_vertical"] else shape_params["md"]
        gxFFT2, gyFFT2 = np.fft.fft2(gx, (md, nd)), np.fft.fft2(gy, (md, nd))
        DGaussxx = np.zeros((len(angleOffset), 1, KernelSize, KernelSize))
        DGaussxy = np.zeros((len(angleOffset), 1, KernelSize, KernelSize))
        DGaussyy = np.zeros((len(angleOffset), 1, KernelSize, KernelSize))
        for i, A in enumerate(angleOffset):
            a, b, c, d = (
                math.cos(-A / 180 * math.pi),
                math.sin(-A / 180 * math.pi),
                math.cos(-A / 180 * math.pi + math.pi / 2),
                math.sin(-A / 180 * math.pi + math.pi / 2),
            )
            DGaussxx[i : i + 1] = np.fft.ifft2(
                (a * gxFFT2 + b * gyFFT2) * (a * gxFFT2 + b * gyFFT2), axes=(-2, -1)
            ).real[None, None, :KernelSize, :KernelSize]
            DGaussyy[i : i + 1] = np.fft.ifft2(
                (c * gxFFT2 + d * gyFFT2) * (c * gxFFT2 + d * gyFFT2), axes=(-2, -1)
            ).real[None, None, :KernelSize, :KernelSize]
            DGaussxy[i : i + 1] = np.fft.ifft2(
                (c * gxFFT2 + d * gyFFT2) * (a * gxFFT2 + b * gyFFT2), axes=(-2, -1)
            ).real[None, None, :KernelSize, :KernelSize]
        DGaussxx = DGaussxx - DGaussxx.mean(axis=(-2, -1), keepdims=True)
        DGaussxy = DGaussxy - DGaussxy.mean(axis=(-2, -1), keepdims=True)
        DGaussyy = DGaussyy - DGaussyy.mean(axis=(-2, -1), keepdims=True)
        return (
            torch.from_numpy(DGaussxx),
            torch.from_numpy(DGaussyy),
            torch.from_numpy(DGaussxy),
        )

    def rotatableKernel(
        self,
        Wsize,
        sigma,
    ):
        k = np.linspace(-Wsize, Wsize, 2 * Wsize + 1)[None, :]
        g = np.exp(-(k**2) / (2 * sigma**2))
        gp = -(k / sigma) * np.exp(-(k**2) / (2 * sigma**2))
        return g.T * gp, gp.T * g

    def TotalVariationLoss(
        self,
        x,
        target,
        Dx,
        Dy,
        mask,
        ind,
    ):
        return (
            torch.abs(
                torch.take_along_dim(
                    torch.conv2d(
                        F.pad(x, self.p_tv, "reflect"),
                        Dx,
                        stride=1,
                        padding=0,
                    ),
                    ind,
                    1,
                )
            )
            * mask
        ).sum() + mask.sum() / mask.numel() * torch.abs(
            torch.take_along_dim(
                torch.conv2d(
                    F.pad(x - target, self.p_tv, "reflect"),
                    Dy,
                    stride=1,
                    padding=0,
                ),
                ind,
                1,
            )
        ).sum()

    def TotalVariationLoss_plain(
        self,
        x,
        target,
        Dx,
        Dy,
        mask,
        ind,
    ):
        return (
            torch.abs(
                torch.conv2d(
                    F.pad(x, self.p_tv, "reflect"),
                    Dx,
                    stride=1,
                    padding=0,
                ),
            )
            * mask
        ).sum() + mask.sum() / mask.numel() * torch.abs(
            torch.conv2d(
                F.pad(x - target, self.p_tv, "reflect"),
                Dy,
                stride=1,
                padding=0,
            ),
        ).sum()

    def HessianRegularizationLoss(
        self,
        x,
        target,
        DGaussxx,
        DGaussyy,
        DGaussxy,
        mask,
        ind,
    ):
        return (
            (
                torch.abs(
                    torch.take_along_dim(
                        torch.conv2d(
                            F.pad(x, self.p_hessian, "reflect"),
                            DGaussxx,
                            stride=1,
                            padding=0,
                        ),
                        ind,
                        1,
                    )
                )
                * mask
            ).sum()
            + mask.sum()
            / mask.numel()
            * torch.abs(
                torch.take_along_dim(
                    torch.conv2d(
                        F.pad(x - target, self.p_hessian, "reflect"),
                        DGaussyy,
                        stride=1,
                        padding=0,
                    ),
                    ind,
                    1,
                )
            ).sum()
            + 2
            * mask.sum()
            / mask.numel()
            * torch.abs(
                torch.take_along_dim(
                    torch.conv2d(
                        F.pad(x - target, self.p_hessian, "reflect"),
                        DGaussxy,
                        stride=1,
                        padding=0,
                    ),
                    ind,
                    1,
                )
            ).sum()
        )

    def HessianRegularizationLoss_plain(
        self,
        x,
        target,
        DGaussxx,
        DGaussyy,
        DGaussxy,
        mask,
        ind,
    ):
        return (
            (
                torch.abs(
                    torch.conv2d(
                        F.pad(x, self.p_hessian, "reflect"),
                        DGaussxx,
                        stride=1,
                        padding=0,
                    ),
                )
                * mask
            ).sum()
            + mask.sum()
            / mask.numel()
            * torch.abs(
                torch.conv2d(
                    F.pad(x - target, self.p_hessian, "reflect"),
                    DGaussyy,
                    stride=1,
                    padding=0,
                ),
            ).sum()
            + 2
            * mask.sum()
            / mask.numel()
            * torch.abs(
                torch.conv2d(
                    F.pad(x - target, self.p_hessian, "reflect"),
                    DGaussxy,
                    stride=1,
                    padding=0,
                ),
            ).sum()
        )

    def forward(
        self,
        network,
        inputs,
        targets,
        mask_dict,
        hy,
        targets_f,
    ):
        outputGNNraw_original, output_att = network(**inputs)
        outputGNNraw_full = (
            F.grid_sample(
                outputGNNraw_original - targets,
                mask_dict["coor"],
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )
            + hy
        )
        outputGNNraw = F.interpolate(
            outputGNNraw_full,
            targets.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        outputGNNraw_original_pad = F.pad(
            targets - outputGNNraw_original,
            (self.GF_pad // 2, self.GF_pad // 2, 0, 0),
            "reflect",
        )

        m, n = outputGNNraw_original.shape[-2:]

        mse = 1 * torch.sum(
            torch.abs(
                outputGNNraw_original_pad[
                    0,
                    0,
                    torch.arange(outputGNNraw_original.shape[-2])[
                        None, None, :, None
                    ].to(outputGNNraw_original_pad.device),
                    mask_dict["ind"],
                ]
            )
        ) + self.lambda_masking_mse * torch.sum(
            torch.abs(targets - outputGNNraw_original) * mask_dict["mse_mask"]
        )

        kernel = pywt.Wavelet("db4")
        l = 6

        output_att_dict = [ptwt.wavedec2(output_att, kernel, level=1, mode="reflect")]

        for _ in range(l - 1):
            output_att_dict.append(
                ptwt.wavedec2(output_att_dict[-1][0], kernel, level=1, mode="reflect")
            )

        output_gnn_hr_dict = [
            ptwt.wavedec2(outputGNNraw_full, kernel, level=1, mode="reflect")
        ]
        for _ in range(l - 1):
            output_gnn_hr_dict.append(
                ptwt.wavedec2(
                    output_gnn_hr_dict[-1][0],
                    kernel,
                    level=1,
                    mode="reflect",
                )
            )

        for i in range(l):
            mse = mse + 20 * torch.sum(
                torch.abs(output_att_dict[i][1][0] - output_gnn_hr_dict[i][1][0])
            )
            mse = mse + 20 * torch.sum(
                torch.abs(output_att_dict[i][1][1] - output_gnn_hr_dict[i][1][1])
            )
            mse = mse + 40 * torch.sum(
                torch.abs(output_att_dict[i][1][2] - output_gnn_hr_dict[i][1][2])
            )

        outputGNNraw_original = torch.cat((outputGNNraw_original, outputGNNraw), 0)

        outputGNNraw_original_f = F.interpolate(
            outputGNNraw_original,
            (
                m,
                n // self.r,
            ),
            mode="bilinear",
            align_corners=True,
        )
        tv = self.total_variation_cal(
            outputGNNraw_original[1:2],
            targets,
            self.Dx,
            self.Dy,
            mask_dict["mask_tv"],
            mask_dict["ind_tv"],
        )

        tv = tv + self.total_variation_cal(
            outputGNNraw_original_f[1:2],
            targets_f,
            self.Dx_f,
            self.Dy_f,
            mask_dict["mask_tv_f"],
            mask_dict["ind_tv_f"],
        )

        hessian = self.HessianRegularizationLoss(
            outputGNNraw_original[1:2],
            targets,
            self.DGaussxx,
            self.DGaussyy,
            self.DGaussxy,
            mask_dict["mask_hessian"],
            mask_dict["ind_hessian"],
        )

        hessian = hessian + self.hessian_cal(
            outputGNNraw_original_f[1:2],
            targets_f,
            self.DGaussxx_f,
            self.DGaussyy_f,
            self.DGaussxy_f,
            mask_dict["mask_hessian_f"],
            mask_dict["ind_hessian_f"],
        )

        return (
            mse + self.lambda_tv * tv + self.lambda_hessian * hessian,
            outputGNNraw_original[0:1],
        )
