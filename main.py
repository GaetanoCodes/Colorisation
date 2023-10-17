from Video.utils import Video, build_video, DVP

if __name__ == "__main__":
    path = "Video/videos/poisson_court2.mp4"
    video = DVP(path)
    video.build_output_video()
    # video.plot_images()
    # build_video(video.video_rgb_1)
