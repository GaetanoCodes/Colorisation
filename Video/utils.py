import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Video:
    def __init__(self, path):
        self.path = path
        self.video, self.fps = self.path_to_tensor_and_fps()
        self.image_number = self.video.shape[0]
        self.video_norm = self.normalize_video_to_100()
        self.video_resized = self.resize_video()

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
        return frames, fps

    def normalize_video_to_100(self, original_max=255):
        video_norm = (100 / original_max) * self.video
        return video_norm

    def resize_video(self):
        resized_video = F.interpolate(
            self.video_norm.unsqueeze(0),
            size=(256, 256),
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
