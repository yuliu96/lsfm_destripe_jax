import numpy as np
import jax
from lsfm_destripe.utils_dual_view import fusion_perslice
from typing import Union, Tuple, Optional, List, Dict
import copy
import jax.numpy as jnp
from lsfm_destripe.utils_jax import (
    transform_cmplx_haiku_model,
    initialize_cmplx_haiku_model,
    update_jax,
    generate_upsample_matrix,
    generate_mask_dict,
)
import tqdm
import dask.array as da
from lsfm_destripe.utils import (
    prepare_aux,
    global_correction,
    destripe_train_params,
)
import matplotlib.pyplot as plt
from lsfm_destripe.network import DeStripeModel
from lsfm_destripe.guided_filter_upsample import GuidedFilterHR_fast
import dask.array as da
from aicsimageio import AICSImage
import tifffile
from lsfm_destripe.loss_term import Loss
import torch


class DeStripe:
    def __init__(
        self,
        resample_ratio: int = 3,
        gf_kernel_size: int = 29,
        hessian_kernel_sigma: float = 1,
        sampling_in_MSEloss: int = 2,
        lambda_tv: float = 1,
        lambda_hessian: float = 1,
        inc: int = 16,
        fast_mode: bool = False,
        n_epochs: int = 300,
        wedge_degree: float = 29,
        n_neighbors: int = 16,
        require_global_correction: bool = True,
        fusion_kernel_size: int = 49,
        fidelity_first: bool = False,
        backend: str = "jax",
        device: str = None,
    ):
        self.train_params = {
            "fast_mode": fast_mode,
            "gf_kernel_size": gf_kernel_size,
            "n_neighbors": n_neighbors,
            "inc": inc,
            "hessian_kernel_sigma": hessian_kernel_sigma,
            "lambda_tv": lambda_tv,
            "lambda_hessian": lambda_hessian,
            "sampling_in_MSEloss": sampling_in_MSEloss,
            "resample_ratio": resample_ratio,
            "n_epochs": n_epochs,
            "wedge_degree": wedge_degree,
            "fusion_kernel_size": fusion_kernel_size,
            "require_global_correction": require_global_correction,
            "fidelity_first": fidelity_first,
        }
        if backend == "torch":
            if device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = device
        else:
            self.device = device
        self.backend = backend

    @staticmethod
    def train_on_one_slice(
        network,
        GuidedFilterHRModel,
        update_method,
        sample_params: Dict,
        train_params: Dict,
        X: np.ndarray,
        mask: np.ndarray = None,
        boundary: np.ndarray = None,
        s_: int = 1,
        z: int = 1,
        backend: str = "jax",
    ):
        rng_seq = jax.random.PRNGKey(0)
        md = (
            sample_params["md"] if sample_params["is_vertical"] else sample_params["nd"]
        )
        nd = (
            sample_params["nd"] if sample_params["is_vertical"] else sample_params["md"]
        )

        if sample_params["view_num"] > 1:
            assert X.shape[1] == 2, print("input X must have 2 channels.")
            assert isinstance(boundary, np.ndarray), print(
                "dual-view fusion boundary is missing."
            )
            dualtarget, fusion_mask = fusion_perslice(
                X[:, :1, :, :],
                X[:, 1:, :, :],
                boundary,
                train_params["fusion_kernel_size"],
            )
        Xd = X[:, :, :: sample_params["r"], :]

        if sample_params["view_num"] > 1:
            dualtargetd, fusion_maskd = fusion_perslice(
                Xd[:, :1, :, :],
                Xd[:, 1:, :, :],
                boundary / sample_params["r"],
                (train_params["fusion_kernel_size"] // sample_params["r"]) // 2 * 2 + 1,
            )

        coor = generate_upsample_matrix(Xd, X, sample_params["r"])

        if sample_params["view_num"] > 1:
            pass
        else:
            dualtarget = copy.deepcopy(X)
            dualtargetd = copy.deepcopy(Xd)
            fusion_maskd = None

        # to Fourier
        Xf = (
            jnp.fft.fftshift(jnp.fft.fft2(Xd), axes=(-2, -1))
            .reshape(1, Xd.shape[1], -1)[0]
            .transpose(1, 0)[: md * nd // 2, :][..., None]
        )

        # initialize
        aver = Xd.sum((2, 3))
        net_params = initialize_cmplx_haiku_model(
            network,
            rng_seq,
            {
                "aver": aver,
                "Xf": Xf,
                "target": dualtargetd,
                "fusion_mask": fusion_maskd,
            },
        )

        opt_state = update_method.opt_init(net_params)

        mask_dict = generate_mask_dict(
            dualtargetd,
            boundary,
            update_method.loss.Dx,
            update_method.loss.Dy,
            update_method.loss.DGaussxx,
            update_method.loss.DGaussyy,
            update_method.loss.p_tv,
            update_method.loss.p_hessian,
            train_params,
            sample_params,
        )

        targets_f = jax.image.resize(
            dualtargetd,
            (1, 1, md, nd // sample_params["r"]),
            method="bilinear",
        )
        mask_dict.update(
            {
                "coor": coor,
            }
        )

        for epoch in tqdm.tqdm(
            range(train_params["n_epochs"]),
            leave=False,
            desc="for {} ({} slices in total): ".format(s_, z),
        ):
            l, net_params, opt_state, Y_raw = update_method(
                epoch,
                net_params,
                opt_state,
                aver,
                Xf,
                fusion_maskd,
                dualtargetd,
                mask_dict,
                dualtarget,
                targets_f,
            )

        Y = 10 ** GuidedFilterHRModel(
            dualtargetd,
            Y_raw,
            coor,
            dualtarget,
            train_params["fidelity_first"],
        )
        return Y[0, 0], 10 ** dualtarget[0, 0]

    @staticmethod
    def train_on_full_arr(
        X: Union[np.ndarray, da.core.Array],
        is_vertical: bool,
        angle_offset_X1: List,
        angle_offset_X2: List,
        mask: Union[np.ndarray, da.core.Array],
        train_params: Dict = None,
        boundary: np.ndarray = None,
        display: bool = False,
        device: str = "cpu",
        non_positive: bool = False,
        backend: str = "jax",
    ):
        if train_params is None:
            train_params = destripe_train_params()
        else:
            train_params = destripe_train_params(**train_params)
        angle_offset = angle_offset_X1 + angle_offset_X2
        angle_offset = list(set(angle_offset))
        r = copy.deepcopy(train_params["resample_ratio"])
        sample_params = {
            "is_vertical": is_vertical,
            "angle_offset": angle_offset,
            "angle_offset_X1": angle_offset_X1,
            "angle_offset_X2": angle_offset_X2,
            "r": r,
        }
        z, view_num, m, n = X.shape
        result = np.zeros((z, m, n), dtype=np.uint16)
        mean = np.zeros(z)
        if sample_params["is_vertical"]:
            n = n if n % 2 == 1 else n - 1
            m = m // train_params["resample_ratio"]
            if m % 2 == 0:
                m = m - 1
            m = m * train_params["resample_ratio"]
        else:
            m = m if m % 2 == 1 else m - 1
            n = n // train_params["resample_ratio"]
            if n % 2 == 0:
                n = n - 1
            n = n * train_params["resample_ratio"]
        sample_params["view_num"] = view_num
        sample_params["m"], sample_params["n"] = m, n
        if sample_params["is_vertical"]:
            sample_params["md"], sample_params["nd"] = (
                m // train_params["resample_ratio"],
                n,
            )
        else:
            sample_params["md"], sample_params["nd"] = (
                m,
                n // train_params["resample_ratio"],
            )
        hier_mask_arr, hier_ind_arr, NI_arr, NI = prepare_aux(
            sample_params["md"],
            sample_params["nd"],
            train_params["fast_mode"],
            sample_params["is_vertical"],
            np.rad2deg(
                np.arctan(r * np.tan(np.deg2rad(sample_params["angle_offset_X1"])))
            ),
            train_params["wedge_degree"],
            train_params["n_neighbors"],
        )
        print("Please check the orientation of the stripes...")
        print(
            "Please check the illumination orientations of the inputs (for dual inputs only)..."
        )
        fig, ax = plt.subplots(1, 2, dpi=200)
        if X.shape[1] > 1:
            title = ["input with top/left illu", "input with bottom/right illu"]
        else:
            title = ["input"]
            ax[1].set_visible(False)
        for i in range(X.shape[1]):
            demo_img = X[X.shape[0] // 2, i, :, :]
            if not sample_params["is_vertical"]:
                demo_img = demo_img.T
            demo_m, demo_n = demo_img.shape
            ax[i].imshow(demo_img)
            for deg in sample_params["angle_offset_X{}".format(i + 1)]:
                d = np.tan(np.deg2rad(deg)) * demo_m
                p0 = [0 + demo_n // 2 - d // 2, d + demo_n // 2 - d // 2]
                p1 = [0, demo_m - 1]
                ax[i].plot(p0, p1, "r")
            ax[i].set_title(title[i], fontsize=8, pad=1)
            ax[i].axis("off")
        plt.show()
        if view_num > 1:
            hier_mask_arr_X2, hier_ind_arr_X2, NI_arr_X2, _ = prepare_aux(
                sample_params["md"],
                sample_params["nd"],
                train_params["fast_mode"],
                sample_params["is_vertical"],
                np.rad2deg(
                    np.arctan(r * np.tan(np.deg2rad(sample_params["angle_offset_X2"])))
                ),
                train_params["wedge_degree"],
                train_params["n_neighbors"],
                NI,
            )
            hier_mask_arr = [hier_mask_arr, hier_mask_arr_X2]
            hier_ind_arr = [hier_ind_arr, hier_ind_arr_X2]
            NI_arr = [NI_arr, NI_arr_X2]

        GuidedFilterHRModel = GuidedFilterHR_fast(
            rx=train_params["gf_kernel_size"],
            ry=1,
            angleList=sample_params["angle_offset_X1"],
        )

        network = transform_cmplx_haiku_model(
            model=DeStripeModel,
            inc=train_params["inc"],
            m_l=(
                sample_params["md"]
                if sample_params["is_vertical"]
                else sample_params["nd"]
            ),
            n_l=(
                sample_params["nd"]
                if sample_params["is_vertical"]
                else sample_params["md"]
            ),
            Angle=sample_params["angle_offset"],
            NI=NI_arr,
            hier_mask=hier_mask_arr,
            hier_ind=hier_ind_arr,
            viewnum=sample_params["view_num"],
            r=sample_params["r"],
            Angle_X1=sample_params["angle_offset_X1"],
            Angle_X2=sample_params["angle_offset_X2"],
            non_positive=non_positive,
        )

        train_params.update(
            {
                "max_pool_kernel_size": (
                    n // 20 * 2 + 1 if sample_params["is_vertical"] else m // 20 * 2 + 1
                )
            }
        )

        update_method = update_jax(network, Loss(train_params, sample_params), 0.01)

        for i in range(z):
            input = np.log10(
                np.clip(np.asarray(X[i : i + 1])[:, :, :m, :n], 1, None)
            )  # (1, v, m, n)
            mask_slice = np.asarray(mask[i : i + 1, :m, :n])[None]
            if not sample_params["is_vertical"]:
                input = input.transpose(0, 1, 3, 2)
                mask_slice = mask_slice.transpose(0, 1, 3, 2)
            input = jnp.asarray(input)
            mask_slice = jnp.asarray(mask_slice)
            boundary_slice = (
                jnp.asarray(
                    boundary[
                        None,
                        None,
                        i : i + 1,
                        : n if sample_params["is_vertical"] else m,
                    ]
                )
                if boundary is not None
                else None
            )
            Y, dualtarget = DeStripe.train_on_one_slice(
                network,
                GuidedFilterHRModel,
                update_method,
                sample_params,
                train_params,
                input,
                mask_slice,
                boundary_slice,
                i + 1,
                z,
                backend=backend,
            )
            if not sample_params["is_vertical"]:
                Y = Y.T
                dualtarget = dualtarget.T
            # Y = np.clip(Y, 0, 65535)
            if display:
                plt.figure(dpi=300)
                ax = plt.subplot(1, 2, 2)
                plt.imshow(Y, vmin=Y.min(), vmax=Y.max(), cmap="gray")
                ax.set_title("output", fontsize=8, pad=1)
                plt.axis("off")
                ax = plt.subplot(1, 2, 1)
                plt.imshow(dualtarget, vmin=Y.min(), vmax=Y.max(), cmap="gray")
                ax.set_title("input", fontsize=8, pad=1)
                plt.axis("off")
                plt.show()
            result[i:, : Y.shape[0], : Y.shape[1]] = np.clip(Y, 0, 65535).astype(
                np.uint16
            )
            mean[i] = np.mean(result[i] + 0.1)
        if train_params["require_global_correction"] and (z != 1):
            print("global correcting...")
            result = global_correction(mean, result)
        print("Done")
        return result

    def train(
        self,
        X1: Union[str, np.ndarray, da.core.Array],
        is_vertical: bool,
        angle_offset_X1: List,
        X2: Union[str, np.ndarray, da.core.Array] = None,
        angle_offset_X2: List = [],
        mask: Union[str, np.ndarray, da.core.Array] = None,
        boundary: Union[str, np.ndarray] = None,
        display: bool = False,
        non_positive: bool = False,
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
        view_num = X.shape[1]
        z, _, m, n = X.shape
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
        if view_num > 1:
            assert not isinstance(boundary, type(None)), print(
                "dual-view fusion boundary is missing."
            )
            if isinstance(boundary, str):
                boundary = tifffile.imread(boundary)
            assert boundary.shape == (z, n) if is_vertical else (z, m), print(
                "boundary index should be of shape [z_slices, n columns]."
            )
        # training
        out = self.train_on_full_arr(
            X,
            is_vertical,
            angle_offset_X1,
            angle_offset_X2,
            mask_data,
            self.train_params,
            boundary,
            display=display,
            device=self.device,
            non_positive=non_positive,
            backend=self.backend,
        )
        return out
