from Video.utils import Video, build_video, DVP

if __name__ == "__main__":
    path = "Video/videos/Poisson.mp4"
    video = DVP(path, size=(128, 128))
    video.optimize(0.1, 1)
    video.plot_an_image()
    video.build_output_video()
    video.build_target_video()
