"""utils"""

import torch
import torchvision.transforms.functional as F


def resize_image(img, size=(256, 256)):
    """
    Resizes a given image to a specified size.

    Args:
        img (torch.Tensor): Image to resize, shape (C, H, W).
        size (tuple): Target size (H, W).

    Returns:
        torch.Tensor: Resized image.
    """
    # Add a batch dimension (None -> batch dim), resize, and remove
    # the batch dimension after resizing.
    return F.resize(img[None, :], size=size)[0, :]


def upsample(image):
    """
    Upsamples an image using bilinear interpolation.

    Args:
        image (torch.Tensor): Image to upsample (shape (C, H, W)).

    Returns:
        torch.Tensor: Upsampled image.
    """
    # Create an upsampling module with a scale factor of 4 and bilinear interpolation mode.
    up = torch.nn.Upsample(scale_factor=4, mode="bilinear")
    return up(image)  # Return the upsampled image.


def get_params(opt_over, net, net_input, downsampler=None):
    """
    Returns parameters to optimize based on the specified options.

    Args:
        opt_over (str): Comma-separated options, e.g., "net", "input", or "down".
        net (torch.nn.Module): The neural network model.
        net_input (torch.Tensor): Tensor storing the input (e.g., noise `z`).
        downsampler (torch.nn.Module, optional): Downsampling module (if applicable).

    Returns:
        list: List of parameters to optimize.
    """
    opt_over_list = opt_over.split(",")  # Split the options into a list.
    params = []  # Initialize a list to store parameters.

    for opt in opt_over_list:
        if opt == "net":
            # If "net" is specified, add the network parameters to the list.
            params += [x for x in net.parameters()]
        elif opt == "down":
            # If "down" is specified, ensure downsampler is provided and add its parameters.
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == "input":
            # If "input" is specified, enable gradient computation for `net_input` and add it.
            net_input.requires_grad = True
            params += [net_input]
        else:
            # Raise an error for invalid options.
            assert False, "Invalid option provided."

    return params  # Return the list of parameters.
