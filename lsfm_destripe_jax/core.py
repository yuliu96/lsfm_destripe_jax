import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch
import torch.fft as fft
from torch.nn import functional as F
import scipy
import torch.nn as nn
import torchvision
import tqdm
from typing import Union, Tuple, Optional, List, Dict
import dask.array as da
from aicsimageio import AICSImage

from jax import random
from jax import jit
import haiku as hk
import jax
import jax.numpy as jnp
from collections.abc import Iterator
from haiku._src.typing import PRNGKey

from destripe_jax.utils_jax import (
    cADAM,
    transform_cmplx_haiku_model,
    initialize_cmplx_haiku_model,
    update_jax,
)
from destripe_jax.network import DeStripeModel, Loss, GuidedFilterLoss
from destripe_jax.guided_filter_variant import (
    GuidedFilterHR,
    GuidedFilterHR_fast,
    GuidedFilter,
)
from destripe_jax.utils import prepare_aux, global_correction, fusion_perslice


class DeStripe:
    def __init__(
        self,
        is_vertical: bool = True,
        angleOffset: List = [0],
        losseps: float = 10,
        qr: float = 0.5,
        resampleRatio: int = 2,
        KGF: int = 29,
        KGFh: int = 29,
        HKs: float = 0.5,
        sampling_in_MSEloss: int = 2,
        isotropic_hessian: bool = True,
        lambda_tv: float = 1,
        lambda_hessian: float = 1,
        inc: int = 16,
        n_epochs: int = 300,
        deg: float = 29,
        Nneighbors: int = 16,
        fast_GF: bool = False,
        require_global_correction: bool = True,
        GF_kernel_size: int = 49,
        Gaussian_kernel_size: int = 49,
    ):
        self.train_params = {
            "fast_GF": fast_GF,
            "KGF": KGF,
            "KGFh": KGFh,
            "losseps": losseps,
            "Nneighbors": Nneighbors,
            "inc": inc,
            "HKs": HKs,
            "lambda_tv": lambda_tv,
            "lambda_hessian": lambda_hessian,
            "sampling": sampling_in_MSEloss,
            "resampleRatio": [resampleRatio, resampleRatio],
            "f": isotropic_hessian,
            "n_epochs": n_epochs,
            "deg": deg,
            "qr": qr,
            "GFr": GF_kernel_size,
            "Gaussianr": Gaussian_kernel_size,
            "angleOffset": angleOffset,
        }
        self.sample_params = {
            "require_global_correction": require_global_correction,
            "is_vertical": is_vertical,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def train_on_one_slice(
        network,
        GuidedFilterHRModel,
        update_method,
        rng_seq: Iterator[PRNGKey],
        sample_params: Dict,
        train_params: Dict,
        X: np.ndarray,
        map: np.ndarray = None,
        dualtarget: np.ndarray = None,
        boundary: np.ndarray = None,
        s_: int = 1,
        z: int = 1,
        device: str = "cuda",
    ):
        md = (
            sample_params["md"] if sample_params["is_vertical"] else sample_params["nd"]
        )
        nd = (
            sample_params["nd"] if sample_params["is_vertical"] else sample_params["md"]
        )
        # put on cuda
        X, map = jnp.array(X), jnp.array(map)
        if sample_params["view_num"] > 1:
            assert X.shape[1] == 2, print("input X must have 2 channels")
            assert isinstance(boundary, np.ndarray), print(
                "dual-view fusion boundary is missing"
            )
            assert isinstance(dualtarget, np.ndarray), print(
                "dual-view fusion result is missing"
            )
            dualtarget = jnp.array(dualtarget)
            boundary = jnp.array(boundary)
        # downsample
        Xd = []
        for ind in range(X.shape[1]):
            Xd.append(
                jax.image.resize(
                    X[:, ind : ind + 1, :, :], (1, 1, md, nd), method="bilinear"
                )
            )
        Xd = jnp.concatenate(Xd, 1)
        if sample_params["view_num"] > 1:
            dualtargetd = jax.image.resize(
                dualtarget, (1, 1, md, nd), method="bilinear"
            )
        map = jax.image.resize(map, (1, 1, md, nd), method="bilinear")
        map = (map > 0).astype(jnp.float32)
        # to Fourier
        Xf = (
            jnp.fft.fftshift(jnp.fft.fft2(Xd))
            .reshape(1, Xd.shape[1], -1)[0]
            .transpose(1, 0)[: md * nd // 2, :]
        )
        # initialize
        _net_params, _net_state = initialize_cmplx_haiku_model(
            network,
            {
                "X": Xd,
                "Xf": Xf,
                "target": Xd if sample_params["view_num"] == 1 else dualtargetd,
                "boundary": boundary,
            },
        )
        opt_init, _, _ = cADAM(0.01)
        _opt_state = opt_init(_net_params)
        smoothedTarget = GuidedFilterLoss(
            r=train_params["KGF"], eps=train_params["losseps"]
        )(Xd, Xd)
        for epoch in tqdm.tqdm(
            range(train_params["n_epochs"]),
            leave=False,
            desc="for {} ({} slices in total): ".format(s_, z),
        ):
            _net_params, _opt_state, _net_state, Y_raw, Y_GNN, Y_LR = update_method(
                epoch,
                _net_params,
                _opt_state,
                Xd,
                Xf,
                boundary,
                Xd if sample_params["view_num"] == 1 else dualtargetd,
                smoothedTarget,
                map,
                next(rng_seq),
                _net_state,
            )
        with torch.no_grad():
            m, n = X.shape[-2:]
            if train_params["fast_GF"] == False:
                resultslice = np.zeros_like(X)
                for index in range(X.shape[1]):
                    input2 = X[:, index : index + 1, :, :]
                    input1 = jax.image.resize(
                        Y_raw[:, index : index + 1, :, :],
                        (1, 1, m, n),
                        method="bilinear",
                    )
                    input1, input2 = torch.tensor(np.asarray(input1)).to(
                        device
                    ), torch.tensor(np.asarray(input2)).to(device)
                    resultslice[:, index : index + 1, :, :] = (
                        10
                        ** GuidedFilterHRModel(input2, input1, r=train_params["qr"])
                        .cpu()
                        .data.numpy()
                    )
                if X.shape[1] > 1:
                    kernel = torch.ones(
                        1, 1, train_params["Gaussianr"], train_params["Gaussianr"]
                    ).to(device) / (train_params["Gaussianr"] ** 2)
                    Y = fusion_perslice(
                        GuidedFilter(r=train_params["GFr"], eps=1),
                        GuidedFilter(r=9, eps=1e-6),
                        resultslice[:, :1, :, :],
                        resultslice[:, 1:, :, :],
                        train_params["Gaussianr"],
                        kernel,
                        torch.tensor(np.asarray(boundary)).to(device),
                        device=device,
                    )
                else:
                    Y = resultslice[0, 0]
            else:
                Y = (
                    10
                    ** GuidedFilterHRModel(
                        (
                            torch.tensor(np.asarray(Xd)).to(device)
                            if sample_params["view_num"] == 1
                            else torch.tensor(np.asarray(dualtargetd)).to(device)
                        ),
                        torch.tensor(np.asarray(Y_GNN)).to(device),
                        (
                            torch.tensor(np.asarray(X)).to(device)
                            if sample_params["view_num"] == 1
                            else torch.tensor(np.asarray(dualtarget)).to(device)
                        ),
                    )
                    .cpu()
                    .data.numpy()[0, 0]
                )
            return Y, resultslice[0] if sample_params["view_num"] > 1 else None

    @staticmethod
    def train_on_full_arr(
        X: Union[np.ndarray, da.core.Array],
        sample_params: Dict,
        train_params: Dict,
        mask: Union[np.ndarray, da.core.Array] = None,
        dualtarget: Union[np.ndarray, da.core.Array] = None,
        boundary: np.ndarray = None,
        display: bool = False,
        device: str = "cpu",
    ):
        rng_seq = hk.PRNGSequence(random.PRNGKey(0))
        z, _, m, n = X.shape
        result = np.zeros((z, m, n), dtype=np.uint16)
        mean = np.zeros(z)
        result_view1, result_view2 = None, None
        if sample_params["view_num"] > 1:
            result_view1, result_view2 = np.zeros((z, m, n), dtype=np.uint16), np.zeros(
                (z, m, n), dtype=np.uint16
            )
            mean_view1, mean_view2 = np.zeros(z), np.zeros(z)
        if train_params["fast_GF"]:
            GuidedFilterHRModel = GuidedFilterHR_fast(
                rx=train_params["KGFh"],
                ry=0,
                angleList=train_params["angleOffset"],
                eps=1e-9,
            ).to(device)
        else:
            GuidedFilterHRModel = GuidedFilterHR(
                rX=[train_params["KGFh"] * 2 + 1, train_params["KGFh"]],
                rY=[0, 0],
                m=(
                    sample_params["m"]
                    if sample_params["is_vertical"]
                    else sample_params["n"]
                ),
                n=(
                    sample_params["n"]
                    if sample_params["is_vertical"]
                    else sample_params["m"]
                ),
                Angle=train_params["angleOffset"],
            ).to(device)
        hier_mask, hier_ind, NI = (
            jnp.asarray(train_params["hier_mask"]),
            jnp.asarray(train_params["hier_ind"]),
            jnp.asarray(train_params["NI"]),
        )
        network = transform_cmplx_haiku_model(
            model=DeStripeModel,
            inc=train_params["inc"],
            KS=train_params["KGF"],
            m=(
                sample_params["md"]
                if sample_params["is_vertical"]
                else sample_params["nd"]
            ),
            n=(
                sample_params["nd"]
                if sample_params["is_vertical"]
                else sample_params["md"]
            ),
            resampleRatio=train_params["resampleRatio"][0],
            Angle=train_params["angleOffset"],
            NI=NI,
            hier_mask=hier_mask,
            hier_ind=hier_ind,
            GFr=train_params["GFr"],
            viewnum=sample_params["view_num"],
        )
        update_method = update_jax(network, Loss(train_params, sample_params), 0.01)
        for i in range(z):
            O = np.log10(np.clip(np.asarray(X[i : i + 1]), 1, None))  # (1, v, m, n)
            if sample_params["view_num"] > 1:
                dualtarget_slice = np.log10(
                    np.clip(np.asarray(dualtarget[i : i + 1]), 1, None)
                )[None]
            mask_slice = np.asarray(mask[i : i + 1])[None]
            boundary_slice = (
                boundary[None, None, i : i + 1, :] if boundary is not None else None
            )
            if not sample_params["is_vertical"]:
                O, mask_slice = O.transpose(0, 1, 3, 2), mask_slice.transpose(
                    0, 1, 3, 2
                )
                if sample_params["view_num"] > 1:
                    dualtarget_slice = dualtarget_slice.transpose(0, 1, 3, 2)
            Y, resultslice = DeStripe.train_on_one_slice(
                network,
                GuidedFilterHRModel,
                update_method,
                rng_seq,
                sample_params,
                train_params,
                O,
                mask_slice,
                dualtarget_slice if sample_params["view_num"] > 1 else None,
                boundary_slice,
                i + 1,
                z,
                device=device,
            )
            if not sample_params["is_vertical"]:
                Y = Y.T
                if sample_params["view_num"] > 1:
                    resultslice = resultslice.transpose(0, 2, 1)
            if display:
                plt.figure(dpi=300)
                ax = plt.subplot(1, 2, 2)
                plt.imshow(Y, vmin=10 ** O.min(), vmax=10 ** O.max(), cmap="gray")
                ax.set_title("output", fontsize=8, pad=1)
                plt.axis("off")
                ax = plt.subplot(1, 2, 1)
                plt.imshow(
                    dualtarget[i] if sample_params["view_num"] > 1 else X[i, 0],
                    vmin=10 ** O.min(),
                    vmax=10 ** O.max(),
                    cmap="gray",
                )
                ax.set_title("input", fontsize=8, pad=1)
                plt.axis("off")
                plt.show()
            result[i] = np.clip(Y, 0, 65535).astype(np.uint16)
            mean[i] = np.mean(result[i] + 0.1)
            if sample_params["view_num"] > 1:
                result_view1[i] = np.clip(resultslice[:1, :, :], 0, 65535).astype(
                    np.uint16
                )
                result_view2[i] = np.clip(resultslice[1:, :, :], 0, 65535).astype(
                    np.uint16
                )
                mean_view1[i] = np.mean(result_view1[i] + 0.1)
                mean_view2[i] = np.mean(result_view2[i] + 0.1)
        if sample_params["require_global_correction"] and (z != 1):
            print("global correcting...")
            result = global_correction(mean, result)
            if sample_params["view_num"] > 1:
                result_view1, result_view2 = global_correction(
                    mean_view1, result_view1
                ), global_correction(mean_view2, result_view2)
        return result, result_view1, result_view2

    def train(
        self,
        X1: Union[str, np.ndarray, da.core.Array],
        X2: Union[str, np.ndarray, da.core.Array] = None,
        mask: Union[str, np.ndarray, da.core.Array] = None,
        dualX: Union[str, np.ndarray, da.core.Array] = None,
        boundary: Union[str, np.ndarray] = None,
        display: bool = False,
    ):
        # read in X
        X1_handle = AICSImage(X1)
        X1_data = X1_handle.get_image_dask_data("ZYX", T=0, C=0)
        if X2 is not None:
            X2_handle = AICSImage(X2)
            X2_data = X2_handle.get_image_dask_data("ZYX", T=0, C=0)
        X = (
            da.stack([X1_data, X2_data], 1)
            if X2 is not None
            else da.stack([X1_data], 1)
        )
        self.sample_params["view_num"] = X.shape[1]
        z, _, m, n = X.shape
        md, nd = (
            m // self.train_params["resampleRatio"][0] // 2 * 2 + 1,
            n // self.train_params["resampleRatio"][1] // 2 * 2 + 1,
        )
        self.sample_params["m"], self.sample_params["n"] = m, n
        self.sample_params["md"], self.sample_params["nd"] = md, nd
        # read in mask
        if mask is None:
            mask_data = np.zeros((z, m, n), dtype=bool)
        else:
            mask_handle = AICSImage(mask)
            mask_data = mask_handle.get_image_dask_data("ZYX", T=0, C=0)
            assert mask_data.shape == (z, m, n), print(
                "mask should be of same shape as input volume(s)."
            )
        # read in dual-result, if applicable
        dualtarget = None
        if self.sample_params["view_num"] > 1:
            assert not isinstance(dualX, type(None)), print(
                "dual-view fusion result is missing."
            )
            assert not isinstance(boundary, type(None)), print(
                "dual-view fusion boundary is missing."
            )
            if isinstance(boundary, str):
                boundary = np.load(boundary)
            dualtarget_handle = AICSImage(dualX)
            dualtarget = dualtarget_handle.get_image_dask_data("ZYX", T=0, C=0)
            assert (
                boundary.shape == (z, n)
                if self.sample_params["is_vertical"]
                else (z, m)
            ), print("boundary index should be of shape [z_slices, n columns].")
            assert dualtarget.shape == (z, m, n), print(
                "fusion result should be of same shape as inputs."
            )
        # training
        hier_mask_arr, hier_ind_arr, NI_arr = prepare_aux(
            self.sample_params["md"],
            self.sample_params["nd"],
            self.sample_params["is_vertical"],
            self.train_params["angleOffset"],
            self.train_params["deg"],
            self.train_params["Nneighbors"],
        )
        self.train_params["NI"] = NI_arr
        self.train_params["hier_mask"] = hier_mask_arr
        self.train_params["hier_ind"] = hier_ind_arr
        result, result_view1, result_view2 = self.train_on_full_arr(
            X,
            self.sample_params,
            self.train_params,
            mask_data,
            dualtarget,
            boundary,
            display=display,
            device=self.device,
        )
        return result, result_view1, result_view2
