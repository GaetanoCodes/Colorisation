import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .dip_model.model import UNet


class Video:
    def __init__(self, path, size=(256, 256), GPU=True):
        if GPU:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.float
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.path = path
        self.video, self.fps = self.path_to_tensor_and_fps()
        self.image_number = self.video.shape[0]
        self.video_norm = self.normalize_video_to_100()
        self.size = size
        self.video_resized = self.resize_video()
        if GPU:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.float
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def path_to_tensor_and_fps(self):
        video = cv2.VideoCapture(self.path)
        # Get the FPS of the video
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                frames.append(frame)
            else:
                break
        frames = torch.tensor(np.array(frames))[:, :, :, 0]
        video.release()
        print("Video converted to torch.Tensor : ")
        print(f"    - Number of frames : {frames.shape[0]},")
        print(f"    - FPS : {fps}.")
        return frames.to(self.dev), fps

    def normalize_video_to_100(self, original_max=255):
        video_norm = (100 / original_max) * self.video
        return video_norm

    def resize_video(self):
        resized_video = F.interpolate(
            self.video_norm.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        )
        resized_video = resized_video.squeeze(0)
        return resized_video

    def plot_images(self, num=16):
        size = int(num**0.5)
        rows = []
        for row in range(size):
            list_row_frames = self.video_resized[row * size : (row + 1) * size, :]
            list_stack = []
            for frame in list_row_frames:
                list_stack.append(frame)
            row_images = torch.vstack(tuple(list_row_frames))
            rows.append(row_images)
        rows = torch.hstack(tuple(rows))
        plt.imshow(rows, cmap="gray")
        plt.show()
        return


class DVP(Video):
    def __init__(self, path, GPU=True, size=(256, 256), frame_number=16):
        super().__init__(path, size=size)

        self.unet = UNet(1, 1, width_multiplier=0.5, trilinear=True, use_ds_conv=False)
        if torch.cuda.is_available():
            self.unet.cuda()
        self.size = (256, 256)
        self.loss_fn = torch.nn.MSELoss()
        self.frame_number = frame_number

        self.video_centered = self.video_center(self.video_resized)
        self.target = self.video_centered[:frame_number][None, :][None, :]
        self.input = self.get_input()

    def video_center(self, video_array, direction="center"):
        if direction == "center":
            video_center = video_array / 50 - 1
            return video_center
        else:
            video_decenter = ((video_array + 1) * 50).astype(int)
            return video_decenter

    def closure(self):
        out = self.unet(self.input)
        total_loss = self.loss_fn(out, self.target)
        total_loss.backward()
        return total_loss

    def get_input(self):
        size = self.size
        frame_N = self.frame_number

        input_ = (
            torch.Tensor(np.random.rand(1, 1, frame_N, size[0], size[1]))
            .type(self.dtype)
            .to(self.dev)
        )
        return input_

    def optimize(self, LR, num_iter):
        """Runs optimization loop.

        Args:
            optimizer_type: 'LBFGS' of 'adam'
            parameters: list of Tensors to optimize over
            closure: function, that returns loss variable
            LR: learning rate
            num_iter: number of iterations
        """

        print("Starting optimization with ADAM")
        parameters = get_params("net", self.unet, self.input)
        optimizer = torch.optim.Adam(parameters, lr=LR)

        for j in range(num_iter):
            optimizer.zero_grad()
            self.closure()
            optimizer.step()
            if j % 10 == 0:
                print(f"Step {j}")
            if j % 100 == 0:
                print(j)
                plt.figure(figsize=(24, 80))
                video_list1 = [
                    self.image_center(
                        self.unet(self.input)[0, 0, i, :].cpu().detach().numpy(),
                        direction="decenter",
                    )
                    for i in range(8)
                ]
                video_list2 = [
                    self.image_center(
                        self.unet(self.input)[0, 0, i, :].cpu().detach().numpy(),
                        direction="decenter",
                    )
                    for i in range(8, 16)
                ]
                # video_list3 = [image_center(model(input_)[0,0,i,:].cpu().detach().numpy(), direction = 'decenter') for i in range(16,24)]
                # video_list4 = [image_center(model(input_)[0,0,i,:].cpu().detach().numpy(), direction = 'decenter') for i in range(24,32)]

                video_list1 = np.hstack(tuple(video_list1))
                video_list2 = np.hstack(tuple(video_list2))
                # video_list3 = np.hstack(tuple(video_list3))
                # video_list4 = np.hstack(tuple(video_list4))

                # plt.imshow(np.vstack((video_list1, video_list2, video_list3, video_list4)), cmap = 'gray')
                plt.imshow(np.vstack((video_list1, video_list2)), cmap="gray")
                # plt.imsave(f"drive/MyDrive/Deep Video Prior/Jackie2/{j}.jpg", np.vstack((video_list1, video_list2 , video_list3, video_list4)), cmap = "gray")
                plt.imsave(
                    f"drive/MyDrive/Deep Video Prior/Jackie/{j}.jpg",
                    np.vstack((video_list1, video_list2)),
                    cmap="gray",
                )

                plt.show()


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


def build_video(video_tensor):
    """the input tensor should be bewteen 0 and 100 (scale of pixel) (at least now for luminance)"""
    size = video_tensor.shape[1], video_tensor.shape[2]
    fps = 30
    out = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]), False
    )
    for i in range(video_tensor.shape[0]):
        data = (video_tensor[i, :] * 255 / 100).type(torch.uint8).cpu().detach().numpy()
        print(data.shape, data)
        out.write(data)
    out.release()
