from Video.utils import Video, build_video, DVP

if __name__ == "__main__":
    path = "Video/videos/poisson_court2.mp4"
    video = DVP(path, size=(128, 128))
    video.optimize(0.1, 1)
    video.build_output_video()
