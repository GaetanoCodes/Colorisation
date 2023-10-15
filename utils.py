import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage
from PIL import Image
import torchvision.transforms.functional as F


def resize_video(video_array: np.array, size=(128, 128), type_="numpy"):
    """Transforms a (t,X,Y) tensor into a (t,size,size) tensor.

    Args:
        video_array (np.array): tensor of the video (shape : (t,X,Y))
        size (tuple, optional): Target size. Defaults to (128,128).

    Returns:
        _type_: _description_
    """
    array = []
    for frame in video_array:
        resized = cv2.resize(frame, size)
        array.append(resized)

    if type_ == "numpy":
        return np.array(array)
    elif type_ == "torch":
        return torch.tensor(array)


# def resize_image(image: np.array, size=(128, 128), type_="numpy"):
#     """Transforms a (X,Y) tensor into a (size,size) tensor.

#     Args:
#         video_array (np.array): tensor of the video (shape : (X,Y))
#         size (tuple, optional): Target size. Defaults to (128,128).

#     Returns:
#         _type_: _description_
#     """

#     resized = skimage.transform.resize(image, size)
#     if type_ == "numpy":
#         return np.array(resized)
#     elif type_ == "torch":
#         return torch.tensor(resized)


def resize_image(img, size=(256, 256), resample=3):
    
    return F.resize(img[None,:], size = size)[0,:]


def postprocess_tens(tens_orig_l, out_ab, mode="bilinear"):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode="bilinear")
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return skimage.color.lab2rgb(
        out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0))
    )


def plot_image(image, title="", size=(5, 5), cmap=""):
    """
    Display LAB image 1x3xHxL
    """

    plt.figure(figsize=size)
    plt.title(title)
    if cmap:
        plt.imshow(image, cmap=cmap)
    plt.show()


def upsample(image):
    up = torch.nn.Upsample(scale_factor=4, mode="bilinear")
    return up(image)


def get_params(opt_over, net, net_input, downsampler=None):
    """Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    """
    opt_over_list = opt_over.split(",")
    params = []

    for opt in opt_over_list:
        if opt == "net":
            params += [x for x in net.parameters()]
        elif opt == "down":
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == "input":
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, "what is it?"

    return params