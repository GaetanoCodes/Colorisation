"""eccv16"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class BaseColor(nn.Module):
    """
    A PyTorch module for handling normalization and unnormalization of LAB color space components.

    Attributes:
        l_cent (float): The center value for the luminance channel normalization.
        l_norm (float): The normalization factor for the luminance channel.
        ab_norm (float): The normalization factor for the chrominance channels (a and b).

    Methods:
        normalize_l(in_l):
            Normalizes the luminance channel (L) to a range centered around 0.

        unnormalize_l(in_l):
            Restores the luminance channel (L) to its original range.

        normalize_ab(in_ab):
            Normalizes the chrominance channels (a and b) to a range centered around 0.

        unnormalize_ab(in_ab):
            Restores the chrominance channels (a and b) to their original range.

        ab_128_to_01(in_ab):
            Converts chrominance channels (a and b) from the range [-128, 128] to [0, 1].

        ab_01_to_128(in_ab):
            Converts chrominance channels (a and b) from the range [0, 1] to [-128, 128].
    """

    def __init__(self):
        """
        Initializes the BaseColor module with predefined normalization constants.

        - `l_cent` is set to 50 to represent the center of the luminance range.
        - `l_norm` is set to 100 for scaling luminance values.
        - `ab_norm` is set to 128 for scaling chrominance (a and b) values.
        """
        super(BaseColor, self).__init__()
        self.l_cent = 50  # Center for luminance normalization.
        self.l_norm = 100.0  # Scale factor for luminance.
        self.ab_norm = 128.0  # Scale factor for chrominance (a and b).

    def normalize_l(self, in_l):
        """
        Normalizes the luminance channel (L).

        Args:
            in_l (torch.tensor): Input luminance values.

        Returns:
            torch.tensor: Normalized luminance values.
        """
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        """
        Restores the luminance channel (L) to its original range.

        Args:
            in_l (torch.tensor): Normalized luminance values.

        Returns:
            torch.tensor: Unnormalized luminance values.
        """
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        """
        Normalizes the chrominance channels (a and b).

        Args:
            in_ab (torch.tensor): Input chrominance values.

        Returns:
            torch.tensor: Normalized chrominance values.
        """
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        """
        Restores the chrominance channels (a and b) to their original range.

        Args:
            in_ab (torch.tensor): Normalized chrominance values.

        Returns:
            torch.tensor: Unnormalized chrominance values.
        """
        return in_ab * self.ab_norm

    def ab_128_to_01(self, in_ab):
        """
        Converts chrominance channels (a and b) from [-128, 128] to [0, 1].

        Args:
            in_ab (torch.tensor): Chrominance values in the range [-128, 128].

        Returns:
            torch.tensor: Chrominance values in the range [0, 1].
        """
        return (in_ab + self.ab_norm) / (2 * self.ab_norm)

    def ab_01_to_128(self, in_ab):
        """
        Converts chrominance channels (a and b) from [0, 1] to [-128, 128].

        Args:
            in_ab (torch.tensor): Chrominance values in the range [0, 1].

        Returns:
            torch.tensor: Chrominance values in the range [-128, 128].
        """
        return in_ab * (2 * self.ab_norm) - self.ab_norm

    def forward(self, input_l: torch.tensor) -> None:
        """forward"""


class ECCVGenerator(BaseColor):
    """
    A deep neural network for colorization based on ECCV16 architecture.
    This model takes grayscale images as input and predicts chrominance
    (a, b) channels in the LAB color space.

    Attributes:
        model1-model8 (nn.Sequential): Stacked convolutional layers forming the
            core of the network.
        softmax (nn.Softmax): Softmax layer to compute probability distributions
            over color bins.
        model_out (nn.Conv2d): Final layer to map 313 color probabilities
            to chrominance channels (a, b).
        upsample4 (nn.Upsample): Upsampling layer to scale the output to
            match the input resolution.
        out_ab_64_max, out_ab_256_max, out_ab_64_mean, input_l, out_ab_256, out_64, proba_64:
            Various placeholders for intermediate results during forward pass.
    """

    def __init__(self, norm_layer=nn.BatchNorm2d):
        """
        Initializes the ECCVGenerator model with eight sequential blocks of convolutional layers,
        followed by a softmax layer for probability distribution over colors and a mapping layer
        for chrominance prediction.

        Args:
            norm_layer (nn.Module): Normalization layer to be used (default: nn.BatchNorm2d).
        """
        super(ECCVGenerator, self).__init__()

        # Define sequential models for each layer group
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        )

        self.model5 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            norm_layer(512),
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            norm_layer(512),
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        )

        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # Additional layers for output processing
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(
            313, 2, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        # Placeholders for intermediate results
        self.out_ab_64_max = 0
        self.out_ab_256_max = 0
        self.out_ab_64_mean = 0
        self.input_l = 0
        self.out_ab_256 = 0
        self.out_64 = 0
        self.proba_64 = 0

    def forward(self, input_l):
        """
        Forward pass through the network. Processes luminance input to produce chrominance probabilities.

        Args:
            input_l (torch.Tensor): Grayscale input (luminance channel).

        Returns:
            torch.Tensor: Chrominance probabilities (shape depends on the model configuration).
        """
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)

        # Compute probabilities over color bins
        proba_64 = self.softmax(conv8_3)
        return proba_64


def eccv16(pretrained=True):
    """
    Initializes an instance of the ECCVGenerator model. If `pretrained` is True,
    the model is loaded with pre-trained weights from a specified URL.

    Args:
        pretrained (bool): Whether to load the model with pre-trained weights.
                           Default is True.

    Returns:
        ECCVGenerator: An instance of the ECCVGenerator model.
    """
    # Create a new instance of the ECCVGenerator model
    model = ECCVGenerator()

    if pretrained:
        # Load pre-trained weights from a remote URL
        model.load_state_dict(
            model_zoo.load_url(
                "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
                map_location="cpu",  # Load weights onto the CPU by default
                check_hash=True,  # Verify the integrity of the downloaded file
            )
        )

    return model
