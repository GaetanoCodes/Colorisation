import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from IPython import embed
import matplotlib.pyplot as plt
from utils import *
import torchvision.transforms as T


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


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model1 += [
            nn.ReLU(True),
        ]
        model1 += [
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
        ]
        model1 += [
            nn.ReLU(True),
        ]
        model1 += [
            norm_layer(64),
        ]

        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model2 += [
            nn.ReLU(True),
        ]
        model2 += [
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
        ]
        model2 += [
            nn.ReLU(True),
        ]
        model2 += [
            norm_layer(128),
        ]

        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
        ]
        model3 += [
            nn.ReLU(True),
        ]
        model3 += [
            norm_layer(256),
        ]

        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model4 += [
            nn.ReLU(True),
        ]
        model4 += [
            norm_layer(512),
        ]

        model5 = [
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
        ]
        model5 += [
            nn.ReLU(True),
        ]
        model5 += [
            norm_layer(512),
        ]

        model6 = [
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
        ]
        model6 += [
            nn.ReLU(True),
        ]
        model6 += [
            norm_layer(512),
        ]

        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model7 += [
            nn.ReLU(True),
        ]
        model7 += [
            norm_layer(512),
        ]

        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
        ]
        model8 += [
            nn.ReLU(True),
        ]
        model8 += [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model8 += [
            nn.ReLU(True),
        ]
        model8 += [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        model8 += [
            nn.ReLU(True),
        ]

        model8 += [
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(
            313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False
        )
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.out_ab_64_max = 0
        self.out_ab_256_max = 0
        self.out_ab_64_mean = 0
        self.input_l = 0
        self.out_ab_256 = 0
        self.out_64 = 0
        self.proba_64 = 0

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)

        out_reg = self.model_out(self.softmax(conv8_3))
        proba_64 = self.softmax(conv8_3)
        # print(
        #     conv8_3.shape
        # )  # conv8 proba, il faut maintenant les coordonnées de chaque couleur

        # self.proba_64 = self.softmax(conv8_3)
        # self.out_ab_64_mean = 110 * out_reg
        # self.input_l = input_l
        # self.out_ab_256 = F.interpolate(110 * out_reg, size=(256, 256), mode="bilinear")

        # return self.unnormalize_ab(self.upsample4(out_reg))
        return proba_64


def eccv16(pretrained=True):
    model = ECCVGenerator()
    if pretrained:
        import torch.utils.model_zoo as model_zoo

        model.load_state_dict(
            model_zoo.load_url(
                "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
                map_location="cpu",
                check_hash=True,
            )
        )

    return model
