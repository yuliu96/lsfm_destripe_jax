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
        lambda_masking_mse: int = 1,
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
            "lambda_masking_mse": lambda_masking_mse,
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
        fusion_mask: np.ndarray = None,
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

        # Xd = X[:, :, :: sample_params["r"], :]
        Xd = jax.image.resize(X, (1, 1, md, nd), method="lanczos5")
        fusion_maskd = fusion_mask[:, :, :: sample_params["r"], :]

        coor = generate_upsample_matrix(Xd, X, sample_params["r"])

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
                "target": Xd,
            },
        )

        opt_state = update_method.opt_init(net_params)

        mask_dict = generate_mask_dict(
            Xd,
            fusion_maskd,
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
            Xd,
            (1, 1, md, nd // sample_params["r"]),
            method="lanczos5",
        )
        mask_dict.update(
            {
                "coor": coor,
                "mse_mask": mask[:, :, :: sample_params["r"], :],
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
                Xd,
                mask_dict,
                X,
                targets_f,
            )
        Y = GuidedFilterHRModel(
            Xd,
            Y_raw,
            coor,
            X,
            fusion_mask,
            sample_params["angle_offset_individual"],
            train_params["fidelity_first"],
        )
        return Y[0, 0], 10 ** X[0, 0]

    @staticmethod
    def train_on_full_arr(
        X: Union[np.ndarray, da.core.Array],
        is_vertical: bool,
        angle_offset_dict: Dict,
        mask: Union[np.ndarray, da.core.Array],
        train_params: Dict = None,
        fusion_mask: Union[np.ndarray, da.core.Array] = None,
        display: bool = False,
        device: str = "cpu",
        non_positive: bool = False,
        backend: str = "jax",
        flag_compose: bool = False,
    ):
        if train_params is None:
            train_params = destripe_train_params()
        else:
            train_params = destripe_train_params(**train_params)
        angle_offset = []
        for key, item in angle_offset_dict.items():
            angle_offset = angle_offset + item
        angle_offset = list(set(angle_offset))
        angle_offset_individual = []
        if flag_compose:
            for i in range(len(angle_offset_dict)):
                angle_offset_individual.append(
                    angle_offset_dict["angle_offset_{}".format(i)]
                )
        else:
            angle_offset_individual.append(angle_offset_dict["angle_offset"])

        r = copy.deepcopy(train_params["resample_ratio"])
        sample_params = {
            "is_vertical": is_vertical,
            "angle_offset": angle_offset,
            "angle_offset_individual": angle_offset_individual,
            "r": r,
        }
        z, _, m, n = X.shape
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

        hier_mask_arr, hier_ind_arr, NI_arr, _ = prepare_aux(
            sample_params["md"],
            sample_params["nd"],
            train_params["fast_mode"],
            sample_params["is_vertical"],
            np.rad2deg(
                np.arctan(r * np.tan(np.deg2rad(sample_params["angle_offset"])))
            ),
            train_params["wedge_degree"],
            train_params["n_neighbors"],
        )

        print("Please check the orientation of the stripes...")
        fig, ax = plt.subplots(
            1, 2 if not flag_compose else len(angle_offset_individual), dpi=200
        )
        if not flag_compose:
            ax[1].set_visible(False)
        for i in range(len(angle_offset_individual)):
            demo_img = X[X.shape[0] // 2, :, :, :]
            if flag_compose:
                demo_img = demo_img * fusion_mask[X.shape[0] // 2, :, :, :]
            if not sample_params["is_vertical"]:
                demo_img = demo_img.swapaxes(2, 1)
            demo_m, demo_n = demo_img.shape[-2:]
            ax[i].imshow(demo_img[i, :])
            for deg in sample_params["angle_offset_individual"][i]:
                d = np.tan(np.deg2rad(deg)) * demo_m
                p0 = [0 + demo_n // 2 - d // 2, d + demo_n // 2 - d // 2]
                p1 = [0, demo_m - 1]
                ax[i].plot(p0, p1, "r")
            ax[i].axis("off")
        plt.show()

        GuidedFilterHRModel = GuidedFilterHR_fast(
            rx=train_params["gf_kernel_size"],
            ry=3,
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
            r=sample_params["r"],
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
            input = np.log10(np.clip(np.asarray(X[i : i + 1])[:, :, :m, :n], 1, None))
            mask_slice = np.asarray(mask[i : i + 1, :m, :n])[None]
            if flag_compose:
                fusion_mask_slice = np.asarray(fusion_mask[i : i + 1])[:, :, :m, :n]
            else:
                fusion_mask_slice = np.ones(input.shape, dtype=np.float32)

            if not sample_params["is_vertical"]:
                input = input.transpose(0, 1, 3, 2)
                mask_slice = mask_slice.transpose(0, 1, 3, 2)
                fusion_mask_slice = fusion_mask_slice.tranpose(0, 1, 3, 2)

            input = jnp.asarray(input)
            mask_slice = jnp.asarray(mask_slice)
            fusion_mask_slice = jnp.asarray(fusion_mask_slice)

            Y, dualtarget = DeStripe.train_on_one_slice(
                network,
                GuidedFilterHRModel,
                update_method,
                sample_params,
                train_params,
                input,
                mask_slice,
                fusion_mask_slice,
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
        is_vertical: bool,
        x: Union[str, np.ndarray, da.core.Array] = None,
        x_compose: Union[str, np.ndarray, da.core.Array] = None,
        mask: Union[str, np.ndarray, da.core.Array] = None,
        fusion_mask: Union[da.core.Array, np.ndarray] = None,
        display: bool = False,
        non_positive: bool = False,
        **kwargs,
    ):  # angle_offset_X2
        if x is not None:
            print("Start DeStripe...\n")
            flag_compose = False
        elif x_compose is not None:
            print("Start DeStripe-FUSE...\n")
            if fusion_mask is None:
                print("fusion_mask cannot be missing.")
                return
            flag_compose = True
        else:
            print("Input is missing.")
            return

        # read in X
        X_handle = AICSImage(x_compose if flag_compose else x)
        X_data = X_handle.get_image_dask_data("ZYX", T=0, C=0)
        X = da.stack([X_data], 1)
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
        angle_offset_dict = {}
        for key, item in kwargs.items():
            if key.startswith("angle_offset"):
                angle_offset_dict.update({key: item})
        # read in dual-result, if applicable
        if x_compose is not None:
            assert not isinstance(fusion_mask, type(None)), print(
                "fusion mask is missing."
            )
            if fusion_mask.ndim == 3:
                fusion_mask = fusion_mask[None]
            assert (
                (fusion_mask.shape[0] == z)
                and (fusion_mask.shape[2] == m)
                and (fusion_mask.shape[3] == n)
            ), print(
                "fusion mask should be of shape [z_slices, ..., m rows, n columns]."
            )
            assert len(angle_offset_dict) == fusion_mask.shape[1], print(
                "angle offsets should be {} in total.".format(fusion_mask.shape[1])
            )
        # training
        out = self.train_on_full_arr(
            X,
            is_vertical,
            angle_offset_dict,
            mask_data,
            self.train_params,
            fusion_mask,
            display=display,
            device=self.device,
            non_positive=non_positive,
            backend=self.backend,
            flag_compose=flag_compose,
        )
        return out
