from typing import Any
from eccv16 import eccv16
from utils import resize_image, plot_image, upsample, get_params
from dip_models import get_net
from dip_models.downsampler import Downsampler
from coef_chrominance import COEFS
import torch
from eccv16 import BaseColor
from skimage import color
import numpy as np
from copy import deepcopy
import kornia
import matplotlib.pyplot as plt

# TODO : qq chose ne va pas dans la proje ou dans l'affichage de la proj
# prévenir frederic et fabien
GPU = False
if GPU:
    DTYPE = torch.cuda.Floattensor
else:
    DTYPE = torch.float


class Colorizer:
    def __init__(self):
        self.eccv = eccv16(pretrained=True)
        self.colorizer = self.eccv.eval()
        if GPU:
            self.colorizer.cuda()

    def __call__(self, image) -> torch.tensor:
        return self.colorizer(torch.tensor(image))

    def lab2rgb(self, image):
        to_numpy = deepcopy(self.lab_mean_64.detach().numpy())
        color.lab2rgb(to_numpy)


COLORIZER = Colorizer()
BASE_COLOR = BaseColor()


class ECCVImage:
    def __init__(self, black_image):
        global COLORIZER, COEFS

        # Should take an image path instead

        black_image_norm = black_image * (100 / 255)
        self.luminance_256 = np.clip(
            resize_image(black_image_norm, size=(256, 256)), 0, 100
        )  # (1,1,256,256), on clip car en resizant on a parfois une interpolation qui depasse 100 ou 0
        self.luminance_64 = np.clip(
            resize_image(black_image_norm, size=(64, 64)), 0, 100
        )
        # we have to define chrominance channels

        self.process_EECV()

        # TODO : on a récup la distribution de proba grâce à self.out_eccv (recup seulement conv_8) (ne lui injecter que de 256x256)
        #           - pouvoir générer une moyenne par pixel rapidement
        #           - faire la fonction de projection

    def process_EECV(self):
        """Recupère l'output de ECCV et postprocess"""
        self.proba_distrib = COLORIZER(self.luminance_256[None, None, :])
        print(self.proba_distrib.shape)
        self.proba_chrom_norm = torch.einsum("abcd,bn->nbcd", self.proba_distrib, COEFS)
        self.proba_chrom_unnorm = BASE_COLOR.unnormalize_ab(self.proba_chrom_norm)
        self.chrom_mean_unnorm = self.proba_chrom_unnorm.sum(axis=1)  # (2,64,64)

        self.lab_mean_64 = torch.cat(
            (self.luminance_64[None, None, :], self.chrom_mean_unnorm[None, :]),
            dim=1,
        )

        self.rgb_mean = kornia.color.lab_to_rgb(self.lab_mean_64)

        self.output_upsampled = upsample(self.lab_mean_64)
        self.output_upsampled[:, [0], :] = self.luminance_256

    def plot_ECCV(self):
        """Plot l'image output en 256x256"""
        rgb256 = kornia.color.lab_to_rgb(self.output_upsampled)
        plt.imshow(rgb256[0, :].permute(1, 2, 0).detach().numpy())
        plt.show()
        return


class LoriaImageColorization(ECCVImage):
    def __init__(self, black_image):
        super().__init__(black_image)
        # DIP network
        self.dip_net = get_net(
            32,
            "skip",
            "reflection",
            n_channels=3,
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode="bilinear",
        ).type(DTYPE)

        self.downsampler = Downsampler(
            n_planes=3, factor=4, kernel_type="lanczos2", phase=0.5, preserve_size=True
        ).type(DTYPE)

        self.dip_input = (
            torch.tensor(np.random.normal(size=(1, 32, 256, 256))).type(DTYPE).detach()
        )
        self.target_dip = self.get_initialized_image()

        self.loss_fn = torch.nn.MSELoss()

    def closure(self, num_iter_active):
        print(num_iter_active)
        self.out = self.dip_net(self.dip_input)
        out = self.out
        if num_iter_active % 20 == 0:
            # on prend les chrominances
            out_chr = out[:, 1:, :].clone().detach()
            # on les passe de [0,1] à [-128,128]
            out_chr_unnorm = BASE_COLOR.ab_01_to_128(out_chr)
            image_cat = torch.cat(
                (
                    self.luminance_256[None, None, :],
                    out_chr_unnorm,
                ),
                dim=1,
            )
            rgb256 = kornia.color.lab_to_rgb(image_cat)
            plt.imsave(
                f"output/{num_iter_active}.png",
                rgb256[0, :].permute(1, 2, 0).detach().numpy(),
            )
            # projection
            out_64 = self.downsampler(out)
            # print(out_64[0, 1:, :].min(), out_64[0, 1:, :].max())

            out_64[:, 0, :, :] = self.luminance_64[None, :]
            # print(out_64.shape)
            projected = self.projection_chrom(out_64)
            projected[:, 0, :] = 100 * projected[:, 0, :]
            projected[:, 1:, :] = BASE_COLOR.ab_01_to_128(projected[:, 1:, :])
            print(
                "Min and max",
                projected[0, 1:, :].min(),
                projected[0, 1:, :].max(),
                projected[:, 0, :].max(),
            )
            projected_rgb = kornia.color.lab_to_rgb(projected)
            plt.imsave(
                f"output/{num_iter_active}_proj.png",
                projected_rgb[0, :].permute(1, 2, 0).detach().numpy(),
            )
            # target
            target_lab = self.target_dip.clone().detach()
            target_lab[:, 0, :] = 100 * target_lab[:, 0, :]
            target_lab[:, 1:, :] = BASE_COLOR.ab_01_to_128(target_lab[:, 1:, :])
            target_rgb = kornia.color.lab_to_rgb(target_lab)
            plt.imsave(
                f"output/{num_iter_active}_taregt.png",
                target_rgb[0, :].permute(1, 2, 0).detach().numpy(),
            )
            # luminance fixee
            image_cat_lum_fixed = image_cat
            image_cat_lum_fixed[:, 0, :] = 0 * image_cat_lum_fixed[:, 0, :] + 30
            image_cat_lum_fixed_rgb = kornia.color.lab_to_rgb(image_cat_lum_fixed)
            ##
            plt.imsave(
                f"output/{num_iter_active}_lum.png",
                image_cat_lum_fixed_rgb[0, :].permute(1, 2, 0).detach().numpy(),
            )
        total_loss = self.loss_coupled_tv(out)
        print("LOSS", total_loss.item())
        total_loss.backward(retain_graph=True)

        return total_loss

    def optimization(self):
        self.optimize(0.02, 2000)

    def loss_coupled_tv(self, out, gamma=80):
        # Coupled TV
        delta_horiz = (out[:, :, 1:, :] - out[:, :, :-1, :]) ** 2
        delta_horiz = delta_horiz[:, :, :, :-1]
        delta_vert = (out[:, :, :, 1:] - out[:, :, :, :-1]) ** 2
        delta_vert = delta_vert[:, :, :-1, :]

        delta = delta_horiz + delta_vert
        delta[:, 0, :] = gamma * delta[:, 0, :]
        epsilon = 0.00001
        coupled_tv = 0.000005 * torch.sum(torch.pow(epsilon + delta, 0.5))
        # Norme L2
        l2 = self.loss_fn(self.downsampler(out), self.target_dip)
        # norm luminance
        loss_lum = self.loss_fn(self.downsampler(out)[:, 0, :], self.luminance_64 / 100)

        return l2 + loss_lum + coupled_tv

    def projection_chrom(self, image, k=313):
        coefs_to_128 = 0.5 * (COEFS + 1)  # entre 0 et 1
        coefs_a = coefs_to_128[:, 0]
        coefs_b = coefs_to_128[:, 1]
        coefs_a = coefs_a[None, :, None, None] * torch.ones(1, 313, 64, 64)
        coefs_b = coefs_b[None, :, None, None] * torch.ones(1, 313, 64, 64)

        ind_min = torch.argmin(
            torch.pow(coefs_a - image[0, 1, :], 2)
            + torch.pow(coefs_b - image[0, 2, :], 2),
            dim=1,
        )

        projected_chrm_a = torch.gather(coefs_a, 1, ind_min.unsqueeze(2)).squeeze(2)
        projected_chrm_b = torch.gather(coefs_b, 1, ind_min.unsqueeze(2)).squeeze(2)

        projected = torch.ones(1, 3, 64, 64)
        # il ne faut pas denormaliseer
        projected[:, 1, :, :] = projected_chrm_a
        projected[:, 2, :, :] = projected_chrm_b
        projected[0, 0, :, :] = self.luminance_64 / 100

        return projected

    def get_initialized_image(self):
        coefs_to_128 = 0.5 * (COEFS + 1)
        coefs_a = coefs_to_128[:, 0]
        coefs_b = coefs_to_128[:, 1]
        coefs_a = coefs_a[None, :, None, None] * torch.ones(1, 313, 64, 64)
        coefs_b = coefs_b[None, :, None, None] * torch.ones(1, 313, 64, 64)
        ind_max = torch.argmax(self.proba_distrib, axis=1)

        chr_a = torch.gather(coefs_a, 1, ind_max.unsqueeze(2)).squeeze(2)
        chr_b = torch.gather(coefs_b, 1, ind_max.unsqueeze(2)).squeeze(2)
        intitialized = torch.ones(1, 3, 64, 64)
        intitialized[:, 1, :, :] = chr_a
        intitialized[:, 2, :, :] = chr_b
        intitialized[:, 0, :, :] = self.luminance_64[None, :] / 100

        # plt.imshow(intitialized_rgb[0, :].permute(1, 2, 0).detach().numpy())
        # plt.show()
        return intitialized

    def optimize(self, LR, num_iter):
        """Runs optimization loop.

        Args:
            optimizer_type: 'LBFGS' of 'adam'
            parameters: list of Tensors to optimize over
            closure: function, that returns loss variable
            LR: learning rate
            num_iter: number of iterations
        """
        parameters = get_params("net", self.dip_net, self.dip_input)
        print("Starting optimization with ADAM")
        optimizer = torch.optim.Adam(parameters, lr=LR)
        print("Nombre d'itérations total :", num_iter)
        for j in range(num_iter):
            print(j)
            if j < 400 or (j % 200 != 0):
                print("Optimization over theta.")
                optimizer.zero_grad()
                self.closure(j)
                optimizer.step()
            # else:
            #     new_target = self.projection_chrom(self.downsampler(self.out))
            #     new_target[0, 0, :] = self.luminance_64.clone().detach() / 100
            #     self.target_dip = new_target.clone().detach()

        print("Optimzation done.")
