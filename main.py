import numpy as np
import utils
import matplotlib.pyplot as plt
from image_colorization import ECCVImage
from PIL import Image

if __name__ == "__main__":
    image = np.array(Image.open("images/lion_fabien.jpg"))
    colorizer = ECCVImage(image)
    colorizer.plot_ECCV()
