# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from typing import List

import torch

from mridc.collections.common.parts.fft import fft2c, ifft2c


def log_likelihood_gradient(
    eta: torch.Tensor,
    masked_kspace: torch.Tensor,
    sense: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    fft_type: str = "orthogonal",
    fft_dim=None,
    coil_dim=None,
) -> torch.Tensor:
    """
    Computes the gradient of the log-likelihood function.

    Args:
        eta: Initial guess for the reconstruction.
        masked_kspace: Masked k-space data.
        sense: Sensing matrix.
        mask: Mask.
        sigma: Noise level.
        fft_type: Type of FFT to use.
        fft_dim: Dimension to use for the FFT.

    Returns
    -------
    torch.Tensor: Gradient of the log-likelihood function.
    """
    if fft_dim is None:
        fft_dim = [2, 3]

    if coil_dim is None:
        coil_dim = 1

    eta_real, eta_imag = map(lambda x: torch.unsqueeze(x, coil_dim - 1), eta.chunk(2, -1))
    sense_real, sense_imag = sense.chunk(2, -1)

    if eta_real.shape[0] != sense_real.shape[0]:  # 3D
        eta_real = eta_real.permute(1, 0, 2, 3, 4)
        eta_imag = eta_imag.permute(1, 0, 2, 3, 4)

    re_se = eta_real * sense_real - eta_imag * sense_imag
    im_se = eta_real * sense_imag + eta_imag * sense_real

    pred = ifft2c(mask * (fft2c(torch.cat((re_se, im_se), -1), fft_type=fft_type, fft_dim=fft_dim) - masked_kspace), fft_type=fft_type, fft_dim=fft_dim)

    pred_real, pred_imag = pred.chunk(2, -1)

    re_out = torch.sum(pred_real * sense_real + pred_imag * sense_imag, coil_dim) / (sigma**2.0)
    im_out = torch.sum(pred_imag * sense_real - pred_real * sense_imag, coil_dim) / (sigma**2.0)

    eta_real = eta_real.squeeze(0)
    eta_imag = eta_imag.squeeze(0)

    if re_out.dim() != eta_real.dim():  # 3D
        re_out = re_out.unsqueeze(coil_dim)
        im_out = im_out.unsqueeze(coil_dim)
        return torch.cat((eta_real, eta_imag, re_out, im_out), 0).squeeze(-1)

    return torch.cat((eta_real, eta_imag, re_out, im_out), 0).unsqueeze(coil_dim - 1).squeeze(-1)
