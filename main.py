"""main"""

import numpy as np
import torch
from PIL import Image

from src.image_colorization import LoriaImageColorization


def main():
    """main"""
    image = torch.tensor(np.array(Image.open("src/images/lion.jpg")))
    loria_colorizer = LoriaImageColorization(image)
    loria_colorizer.plot_eccv()
    loria_colorizer.optimize(0.02, 5)
    loria_colorizer.plot_result()


if __name__ == "__main__":
    main()
