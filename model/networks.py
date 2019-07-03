"""

"""
from torch import nn
import torch
from torch.nn.utils import spectral_norm

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass


##################################################################################
# Gated conv
##################################################################################
class Gated_conv(nn.Module):
    # mask is binary, 0 is masked point, 1 is not

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        super(Gated_conv, self).__init__()
        # self.relu = nn.ReLU()

        self.gated_conv = spectral_norm(nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride,
                          padding))
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gated_conv(x)
        feature, mask = torch.chunk(x, 2, dim=1)
        x = self.leakyRelu(feature) * torch.sigmoid(mask)
        return x