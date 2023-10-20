import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .dip_model.model import UNet
import kornia
import torch.nn as nn


class BaseColor(nn.Module):
    def __init__(self):
        super(BaseColor, self).__init__()

        self.l_cent = 50
        self.l_norm = 100.0
        self.ab_norm = 128.0

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm

    def ab_128_to_01(self, in_ab):
        return (in_ab + self.ab_norm) / (2 * self.ab_norm)

    def ab_01_to_128(self, in_ab):
        return in_ab * (2 * self.ab_norm) - self.ab_norm


BASE_COLOR = BaseColor()


class Video:
    def __init__(self, path, size=(256, 256), GPU=True, first=5, last=10):
        if GPU:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.float
        self.dev = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.first_unknown = 10
        self.last_unknown = 11
        self.path = path
        self.video_color, self.fps = self.path_to_tensor_and_fps()
        self.video = self.video_color[:, 0, :]
        self.image_number = self.video.shape[0]
        self.size = size
        self.video_resized = self.resize_video()  # color /255 resized
        self.video_rgb_1_resized = self.video_rgb_to_1()
        self.video_lab_128_resized = self.video_lab_to_128()
        self.video_lab_1_resized = self.video_lab_to_1()
        # print(self.video_lab_1_resized.shape)
        # print(
        #     "Minimum of lab 1",
        #     self.video_lab_1_resized.min(),
        #     self.video_lab_1_resized.max(),
        # )

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
        frames = torch.tensor(np.array(frames))
        video.release()
        print("Video converted to torch.Tensor : ")
        print(f"    - Number of frames : {frames.shape[0]},")
        print(f"    - FPS : {fps}.")
        frames = frames.permute(0, 3, 1, 2)
        print(frames.shape)
        return frames.type(self.dtype).to(self.dev), fps

    def normalize_video_to_100(self, original_max=255):
        video_norm = (100 / original_max) * self.video
        return video_norm

    def resize_video(self):
        # revoir la méthode pour qu'il prenne un tenseur 5d
        print(self.video_color.shape)
        print(self.video_color.type())
        video_chr_a = self.video_color[:, 1, :]
        video_chr_b = self.video_color[:, 2, :]
        video_l = self.video_color[:, 0, :]
        resized_l = F.interpolate(
            video_l.unsqueeze(0),
            size=list(self.size),
            mode="bilinear",
            align_corners=False,
        )
        resized_chr_a = F.interpolate(
            video_chr_a.unsqueeze(0),
            size=list(self.size),
            mode="bilinear",
            align_corners=False,
        )
        resized_chr_b = F.interpolate(
            video_chr_b.unsqueeze(0),
            size=list(self.size),
            mode="bilinear",
            align_corners=False,
        )

        resized_video = torch.cat((resized_l, resized_chr_a, resized_chr_b), dim=0)
        resized_video = resized_video.permute(1, 0, 2, 3)
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

    def video_rgb_to_1(self):
        return self.video_resized / 255

    def video_lab_to_128(self):
        lab = kornia.color.rgb_to_lab(self.video_rgb_1_resized)
        return lab

    def video_lab_to_1(self):
        lab_1 = self.video_lab_128_resized.clone()
        lab_1[:, 0, :] = lab_1[:, 0, :] / 100
        lab_1[:, 1:, :] = BASE_COLOR.ab_128_to_01(lab_1[:, 1:, :])
        return lab_1


class DVP(Video):
    def __init__(
        self, path, GPU=False, size=(256, 256), frame_number=16, first=5, last=10
    ):
        super().__init__(path, size=size, GPU=GPU, first=first, last=last)

        self.unet = UNet(3, 3, width_multiplier=0.5, trilinear=True, use_ds_conv=False)
        if torch.cuda.is_available():
            self.unet.cuda()
        self.size = size
        self.loss_fn = torch.nn.MSELoss()
        self.frame_number = frame_number

        self.input = self.get_input()
        self.mask = self.get_mask()
        self.target = self.get_target()

    def get_mask(self):
        mask = torch.ones(
            size=[1, 3, self.frame_number, self.size[0], self.size[1]]
        ).type(torch.bool)
        first_unknown = 5
        last_unknown = 10
        mask[:, 1:, first_unknown:last_unknown, :] = 0
        return mask.to(self.dev)

    def get_target(self, method="propagation") -> torch.Tensor:
        if method == "propagation":
            target = self.video_lab_1_resized[: self.frame_number][None, :].permute(
                0, 2, 1, 3, 4
            )
            target = target * self.mask + (~self.mask) * 0.5
            target = target
            return target.clone()
        else:
            return torch.ones(1)

    def get_input(self):
        size = self.size
        frame_N = self.frame_number

        input_ = (
            torch.Tensor(np.random.rand(1, 3, frame_N, size[0], size[1]))
            .type(self.dtype)
            .to(self.dev)
        )
        return input_

    def output_to_rgb(self, out):
        out_rgb = out.clone()
        out_rgb[:, :, 0, :] = 100 * (out_rgb[:, :, 0, :])
        out_rgb[:, :, 1:, :] = 128 * (2 * out_rgb[:, :, 1:, :] - 1)
        out_rgb = kornia.color.lab_to_rgb(out_rgb)
        out_rgb = 100 * out_rgb
        return out_rgb

    def build_output_video(self):
        out = self.out.clone()
        out = out.permute(0, 2, 1, 3, 4)  # out
        out[0, :, 0, :] = self.video_lab_1_resized[: self.frame_number, 0, :]
        out_rgb = self.output_to_rgb(out) / 100
        build_video(out_rgb, name="output")

        out = self.out.clone()
        out = out.permute(0, 2, 1, 3, 4)  # out
        out[0, :, 0, :] = 0.5  # self.video_lab_1_resized[: self.frame_number, 0, :]
        out_rgb = self.output_to_rgb(out) / 100
        build_video(out_rgb, name="chr_output")

    def build_target_video(self):
        out = self.target.clone()
        out = out.permute(0, 2, 1, 3, 4)  # out
        out_rgb = self.output_to_rgb(out) / 100
        build_video(out_rgb, name="target")

        out = self.target.clone()
        out = out.permute(0, 2, 1, 3, 4)  # out
        out[0, :, 0, :] = 0.5
        out_rgb = self.output_to_rgb(out) / 100
        build_video(out_rgb, name="target_chr")

    def closure(self, method="propagation"):
        if method == "propagation":
            self.out = self.unet(self.input)
            loss_mask = self.loss_fn(
                self.out * self.mask + (~self.mask) * 0.5, self.target
            )
            loss_lum = self.loss_fn(self.out[0, 0, :], self.target[0, 0, :])
            total_loss = loss_mask + loss_lum
            total_loss.backward()
            print("closure", self.out.shape, self.target.shape)
            return

    def plot_an_image(self, frame=5):
        out = self.target.permute(0, 2, 1, 3, 4).clone()  # out
        # print("Min and max of target", out.min(), out.max())

        out_rgb = self.output_to_rgb(out) / 100
        plt.imshow(
            out_rgb[0, frame].permute(1, 2, 0).cpu().detach().numpy(), cmap="gray"
        )
        plt.show()

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
                out = self.out.permute(0, 2, 1, 3, 4).clone()  # out
                # print("Min and max of target", out.min(), out.max())

                out_rgb = self.output_to_rgb(out) / 100
                plt.imshow(
                    out_rgb[0, 0].permute(1, 2, 0).cpu().detach().numpy(), cmap="gray"
                )
                plt.show()

        # output = self.unet(self.input)

        # plt.figure(figsize=(24, 80))
        # video_list1 = [
        #     self.image_center(
        #         output_rgb[0, 0, i, :].cpu().detach().numpy(),
        #         direction="decenter",
        #     )
        #     for i in range(8)
        # ]
        # video_list2 = [
        #     self.image_center(
        #         output_rgb[0, 0, i, :].cpu().detach().numpy(),
        #         direction="decenter",
        #     )
        #     for i in range(8, 16)
        # ]
        # # video_list3 = [image_center(model(input_)[0,0,i,:].cpu().detach().numpy(), direction = 'decenter') for i in range(16,24)]
        # # video_list4 = [image_center(model(input_)[0,0,i,:].cpu().detach().numpy(), direction = 'decenter') for i in range(24,32)]

        # video_list1 = np.hstack(tuple(video_list1))
        # video_list2 = np.hstack(tuple(video_list2))
        # # video_list3 = np.hstack(tuple(video_list3))
        # # video_list4 = np.hstack(tuple(video_list4))

        # # plt.imshow(np.vstack((video_list1, video_list2, video_list3, video_list4)), cmap = 'gray')
        # plt.imshow(np.vstack((video_list1, video_list2)), cmap="gray")
        # # plt.imsave(f"drive/MyDrive/Deep Video Prior/Jackie2/{j}.jpg", np.vstack((video_list1, video_list2 , video_list3, video_list4)), cmap = "gray")
        # plt.imsave(
        #     f"drive/MyDrive/Deep Video Prior/Jackie/{j}.jpg",
        #     np.vstack((video_list1, video_list2)),
        #     cmap="gray",
        # )

        # plt.show()


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


def build_video(video_tensor, name="output"):
    print("Building video.")
    size = video_tensor.shape[3], video_tensor.shape[4]
    fps = 30
    out = cv2.VideoWriter(
        f"{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]), True
    )
    print("Video", video_tensor.min(), video_tensor.max())
    print((video_tensor * 255).shape)
    for i in range(video_tensor.shape[1]):
        data = (
            (video_tensor[0, i, :] * 255)
            .permute(1, 2, 0)
            .type(torch.uint8)
            .cpu()
            .detach()
            .numpy()
        )
        out.write(data)
    out.release()
    print("Video built.")
