# encoding: utf-8
# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI
# Poisson sample function based on https://github.com/rmsouza01/iMRI-challenge

import contextlib
from typing import Optional, Sequence, Tuple, Union, Any

import numpy as np
import torch
from numpy import ndarray


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
                For 2D setting this value corresponds to setting the Full-Width-Half-Maximum.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
    ) -> torch.Tensor:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).
    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have
            at least 3 dimensions. Samples are drawn along the second last
            dimension.
        seed: Seed for the random number generator. Setting the seed
            ensures the same mask is generated each time for the same
            shape. The random state is reset afterwards.
        half_scan_percentage : TODO: implement it for 1D masking
        Returns
        -------
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            mask = self.rng.uniform(size=num_cols) < prob
            pad = torch.div(
                (num_cols - num_low_freqs + 1), 2, rounding_mode="trunc"
            ).item()
            mask[pad : pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.
    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
    ) -> Tuple[torch.Tensor, Union[int, Sequence[int]]]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have
            at least 3 dimensions. Samples are drawn along the second last
            dimension.
        seed: Seed for the random number generator. Setting the seed
            ensures the same mask is generated each time for the same
            shape. The random state is reset afterwards.
        half_scan_percentage : TODO: implement it for 1D masking
        Returns
        -------
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = torch.div(
                (num_cols - num_low_freqs + 1), 2, rounding_mode="trunc"
            ).item()
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
            )
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask, acceleration


class Gaussian2DMaskFunc(MaskFunc):
    def __call__(
        self,
        shape: Union[Sequence[int], ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: float = 0.02,
    ) -> Tuple[Any, int]:
        self.shape = tuple(shape[1:-1])

        full_width_half_maximum, acceleration = self.choose_acceleration()

        if not isinstance(full_width_half_maximum, list):
            full_width_half_maximum = [full_width_half_maximum] * 2
        self.full_width_half_maximum = full_width_half_maximum

        self.acceleration = acceleration
        self.scale = scale

        mask = self.gaussian_kspace()
        mask[tuple(self.gaussian_coordinates())] = 1.0

        if half_scan_percentage != 0:
            mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

        return (
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(-1),
            acceleration,
        )

    def gaussian_kspace(self):
        a, b = self.scale * self.shape[0], self.scale * self.shape[1]
        afocal, bfocal = self.shape[0] / 2, self.shape[1] / 2
        xx, yy = np.mgrid[: self.shape[0], : self.shape[1]]
        ellipse = np.power((xx - afocal) / a, 2) + np.power((yy - bfocal) / b, 2)
        return (ellipse < 1).astype(float)

    def gaussian_coordinates(self):
        n_sample = int(self.shape[0] * self.shape[1] / self.acceleration)
        cartesian_prod = list(np.ndindex(self.shape))
        kernel = self.gaussian_kernel()
        idxs = np.random.choice(
            range(len(cartesian_prod)), size=n_sample, replace=False, p=kernel.flatten()
        )
        return list(zip(*list(map(cartesian_prod.__getitem__, idxs))))

    def gaussian_kernel(self):
        kernels = []
        for fwhm, kern_len in zip(self.full_width_half_maximum, self.shape):
            sigma = fwhm / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x ** 2 / (2 * sigma ** 2)))
            kernels.append(g)
        kernel = np.sqrt(np.outer(kernels[0], kernels[1]))
        kernel = kernel / kernel.sum()
        return kernel


class Gaussian1DMaskFunc(MaskFunc):
    def __call__(
        self,
        shape: Union[Sequence[int], ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: float = 0.02,
    ) -> torch.Tensor:
        self.shape = tuple(shape[1:-1])

        full_width_half_maximum, acceleration = self.choose_acceleration()
        if not isinstance(full_width_half_maximum, list):
            full_width_half_maximum = [full_width_half_maximum] * 2
        self.full_width_half_maximum = full_width_half_maximum
        self.acceleration = acceleration

        self.scale = scale

        mask = self.gaussian_kspace()
        mask[tuple(self.gaussian_coordinates())] = 1.0

        mask = np.fft.ifftshift(
            np.fft.ifftshift(np.fft.ifftshift(mask, axes=0), axes=0), axes=(0, 1)
        )

        if half_scan_percentage != 0:
            mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(-1)

    def gaussian_kspace(self):
        scaled = int(self.shape[0] * self.scale)
        center = np.ones((scaled, self.shape[1]))
        top_scaled = torch.div(
            (self.shape[0] - scaled), 2, rounding_mode="trunc"
        ).item()
        bottom_scaled = self.shape[0] - scaled - top_scaled
        top = np.zeros((top_scaled, self.shape[1]))
        btm = np.zeros((bottom_scaled, self.shape[1]))
        return np.concatenate((top, center, btm))

    def gaussian_coordinates(self):
        n_sample = int(self.shape[0] / self.acceleration)
        kernel = self.gaussian_kernel()
        idxs = np.random.choice(
            range(self.shape[0]), size=n_sample, replace=False, p=kernel
        )
        xsamples = np.concatenate([np.tile(i, self.shape[1]) for i in idxs])
        ysamples = np.concatenate([range(self.shape[1]) for _ in idxs])
        return xsamples, ysamples

    def gaussian_kernel(self):
        kernel = 1
        for fwhm, kern_len in zip(self.full_width_half_maximum, self.shape):
            sigma = fwhm / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x ** 2 / (2 * sigma ** 2)))
            kernel = g
            break
        kernel = kernel / kernel.sum()
        return kernel


def create_mask_for_mask_type(
    mask_type_str: str, center_fractions: Sequence[float], accelerations: Sequence[int]
) -> MaskFunc:
    """
        Creates a mask of the specified type.
    Parameters
    ----------
    mask_type_str :
    center_fractions :  What fraction of the center of k-space to include.
                        For gaussian masking server as Full-Width-at-Half-Maximum.
    accelerations :
    Returns
    -------
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    if mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    if mask_type_str == "gaussian1d":
        return Gaussian1DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "gaussian2d":
        return Gaussian2DMaskFunc(center_fractions, accelerations)
    raise Exception(f"{mask_type_str} not supported")