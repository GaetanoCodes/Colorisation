"""main"""

import numpy as np
import torch
from PIL import Image

from src.image_colorization import LoriaImageColorization


def main():
    """numerical example"""
    lr = 0.02  #############################################################################################
    n_iter = 1000
    image = torch.tensor(np.array(Image.open("src/images/papillon.jpg")))
    loria_colorizer = LoriaImageColorization(image)
    loria_colorizer.plot_eccv()
    loria_colorizer.optimize(lr, n_iter)
    loria_colorizer.plot_result()


if __name__ == "__main__":
    main()
