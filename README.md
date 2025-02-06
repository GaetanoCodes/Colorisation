<p align="center">
  <img src="figures/title.png"   />
</p>

[![Python badge](https://img.shields.io/badge/Python-3.11.11-0066cc?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/downloads/release/python-31111/)
[![Pytorch badge](https://img.shields.io/badge/Pytorch-2.5-cc3300?style=for-the-badge&logo=pytorch)](https://pytorch.org/docs/2.5/)

[![Pylint badge](https://img.shields.io/badge/Linting-pylint-brightgreen?style=for-the-badge)](https://pylint.pycqa.org/en/latest/)
[![Ruff format badge](https://img.shields.io/badge/Formatter-Ruff-000000?style=for-the-badge)](https://docs.astral.sh/ruff/formatter/)

This code allows to reproduce the results of this [paper](https://hal.science/hal-04035467). 
It compares the result given by Zhang *et al.* in the EECV paper [Colorful Image Colorization](https://richzhang.github.io/colorization/).


## Installation
A script is available for an easy creation of the conda environment and compilation of auxiliary functions:
```bash
$ source install.bash
```

## How to use

An example can be found in the `main.py`. 

* Here are the chrominances channel for 2 images (Zhang *et al.* on the left and our method on the right). Out procedure allows recovering sharp contours.

<p align="center">
  <img src="figures/chr_oiseau_comparison.png"   width="600"  />
<!-- </p>
<p align="center"> -->
  <img src="figures/chr_papillon_comparison.png"  width="600"  />
</p>

* In the real images, our algorithm allows correcting halos created in the CNN of Zhang *et al.* and recover more realistic colors contained in the predicted probability distribution of Zhang *et al.*.

<p align="center">
  <img src="figures/lion.png"   width="600"  />
<!-- </p>
<p align="center"> -->
  <img src="figures/meduse.png"  width="600"  />
</p>

## Interesting ? 

If you have any questions, feel free to contact us. We will be more than happy to answer ! 😀

If you use it, a reference to the paper would be highly appreciated.

```
@InProceedings{10.1007/978-3-031-31975-4_23,
author="Agazzotti, Gaetano
and Pierre, Fabien
and Sur, Fr{\'e}d{\'e}ric",
editor="Calatroni, Luca
and Donatelli, Marco
and Morigi, Serena
and Prato, Marco
and Santacesaria, Matteo",
title="Deep Image Prior Regularized by Coupled Total Variation for Image Colorization",
booktitle="Scale Space and Variational Methods in Computer Vision",
year="2023",
publisher="Springer International Publishing",
address="Cham",
pages="301--313",
isbn="978-3-031-31975-4"
}

```


## Tested on

[![Ubuntu badge](https://img.shields.io/badge/Ubuntu-24.04-cc3300?style=for-the-badge&logo=ubuntu)](https://www.releases.ubuntu.com/24.04/)
[![Conda badge](https://img.shields.io/badge/conda-24.9.2-339933?style=for-the-badge&logo=anaconda)](https://docs.conda.io/projects/conda/en/24.9.x/)


[![GPU badge](https://img.shields.io/badge/GPU-T4-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com)
[![Intel badge](https://img.shields.io/badge/CPU-%20Xeon%202.20GHZ-blue?style=for-the-badge&logo=intel)](https://ark.intel.com/content/www/fr/fr/ark/products/196449/intel-core-i7-10510u-processor-8m-cache-up-to-4-90-ghz.html)

