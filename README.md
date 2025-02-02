# Data Consistency for Magnetic Resonance Imaging

[![CodeQL](https://github.com/wdika/mridc/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/wdika/mridc/actions/workflows/codeql-analysis.yml)
[![CircleCI](https://circleci.com/gh/wdika/mridc/tree/main.svg?style=svg)](https://circleci.com/gh/wdika/mridc/tree/main)
[![codecov](https://codecov.io/gh/wdika/mridc/branch/main/graph/badge.svg?token=KPPQ33DOTF)](https://codecov.io/gh/wdika/mridc)
[![DeepSource](https://deepsource.io/gh/wdika/mridc.svg/?label=active+issues&show_trend=true&token=txj87v43GA6vhpbSwPEUTQtX)](https://deepsource.io/gh/wdika/mridc/?ref=repository-badge)
[![DeepSource](https://deepsource.io/gh/wdika/mridc.svg/?label=resolved+issues&show_trend=true&token=txj87v43GA6vhpbSwPEUTQtX)](https://deepsource.io/gh/wdika/mridc/?ref=repository-badge)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/wdika/mridc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/wdika/mridc/alerts/)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

---

❗This is a fork from the original MRIDC repo, expanding on 3 things:

- Made the RIM able to accept 3D data (for a batch size of 1). To use this, use the parameter consec_slices > 1.
- Included 2.5D convolutions for the (CI)RIM. This turns convolutional filter from a single 2D filter to 2 sequential 2D filters, one in-plane and one with the same dimensions in slice dimension and height. To use this feature, set dimensionality to 3.
- Included varialbe density poisson masking as an undersampling technique. To use this, use poisson2d as mask type.

Additionally, this repo contains Thesis material used and described in the paper for reference of implementation, aswell as the Thesis PDF document itself.

A comparison of the Poisson masking compared with conventional Gaussian datamasking:

![alt text](https://github.com/jerke123/mridc/blob/main/thesis%20material/gaussian-poisson-comp.png?raw=true)

(Row A: Poisson undersampled test. Row B: Gaussian undersampled test)

**Data Consistency (DC) is crucial for generalization in multi-modal MRI data and robustness in detecting pathology.**

This repo implements the following reconstruction methods:

- Cascades of Independently Recurrent Inference Machines (CIRIM) [1],
- Independently Recurrent Inference Machines (IRIM) [2, 3],
- End-to-End Variational Network (E2EVN), [4, 5]
- the UNet [5, 6],
- Compressed Sensing (CS) [7], and
- zero-filled reconstruction (ZF).

The CIRIM, the RIM, and the E2EVN target unrolled optimization by gradient descent. Thus, DC is implicitly enforced.
Through cascades DC can be explicitly enforced by a designed term [1, 4].

## Installation

You can install mridc with pip:

### Pip
```bash
pip install mridc
```

### From source
```bash
git clone https://github.com/wdika/mridc
cd mridc
./reinstall.sh
```

## Usage

Recommended public datasets to use with this repo:

- [fastMRI](https://fastmri.org/) [5].

## Documentation

[![Documentation Status](https://readthedocs.org/projects/mridc/badge/?version=latest)](https://mridc.readthedocs.io/en/latest/?badge=latest)

Read the docs [here](https://mridc.readthedocs.io/en/latest/index.html)

## License

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Citation

Check CITATION.cff file or cite using the widget. Alternatively cite as

```BibTeX
@misc{mridc,
    author = {Karkalousos, Dimitrios and Caan, Matthan},
    title = {MRIDC: Data Consistency for Magnetic Resonance Imaging},
    year = {2021},
    url = {https://github.com/wdika/mridc},
}
```

## Bibliography

[1] Karkalousos, D. et al. (2021) ‘Assessment of Data Consistency through Cascades of Independently Recurrent Inference
Machines for fast and robust accelerated MRI reconstruction’. Available at: https://arxiv.org/abs/2111.15498v1 (
Accessed: 1 December 2021).

[2] Lønning, K. et al. (2019) ‘Recurrent inference machines for reconstructing heterogeneous MRI data’, Medical Image
Analysis, 53, pp. 64–78. doi: 10.1016/j.media.2019.01.005.

[3] Karkalousos, D. et al. (2020) ‘Reconstructing unseen modalities and pathology with an efficient Recurrent Inference
Machine’, pp. 1–31. Available at: http://arxiv.org/abs/2012.07819.

[4] Sriram, A. et al. (2020) ‘End-to-End Variational Networks for Accelerated MRI Reconstruction’, Lecture Notes in
Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics),
12262 LNCS, pp. 64–73. doi: 10.1007/978-3-030-59713-9_7.

[5] Zbontar, J. et al. (2018) ‘fastMRI: An Open Dataset and Benchmarks for Accelerated MRI’, arXiv, pp. 1–35. Available
at: http://arxiv.org/abs/1811.08839.

[6] Ronneberger, O., Fischer, P. and Brox, T. (2015) ‘U-Net: Convolutional Networks for Biomedical Image Segmentation’,
in Medical image computing and computer-assisted intervention : MICCAI ... International Conference on Medical Image
Computing and Computer-Assisted Intervention, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28.

[7] Lustig, M. et al. (2008) ‘Compressed Sensing MRI’, IEEE Signal Processing Magazine, 25(2), pp. 72–82. doi:
10.1109/MSP.2007.914728.
