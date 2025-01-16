from src.Video.utils import Video, build_video, DVP

if __name__ == "__main__":
    path = "Colorisation/Video/videos/Poisson.mp4"
    video = DVP(path, size=(128, 128))
    video.optimize(0.1, 1)
    video.plot_an_image()
    video.build_output_video()
    video.build_target_video()
    # TODO:enelever les print, faire un dossier de sauvegarde des outputs avec date et heure
    # le mask ne va pas,  on en fait aussi sur la luminance
    # dans la loss, essayer de contraindre que les chr, i.e.
    # loss_fn(out[1:, list_frames_connues], target[1:, list_frames_connues])
