import numpy as np
import utils
import matplotlib.pyplot as plt
from image_colorization import ECCVImage, LoriaImageColorization
from PIL import Image

if __name__ == "__main__":
    image = np.array(Image.open("images/lion_fabien.jpg"))  # [:, :, 0]

    loria_colorizer = LoriaImageColorization(image)
    # loria_colorizer.plot_ECCV()
    loria_colorizer.optimization()
