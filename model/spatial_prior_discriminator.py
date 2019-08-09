import os

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import scipy.io as sio
import numpy as np

from util.util import weights_init


class SP_Prior_FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(SP_Prior_FCDiscriminator, self).__init__()
        self.foreground_map = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        n = len(self.foreground_map)
        self.gamma = nn.Parameter(torch.zeros(n))
        self.conv1 = spectral_norm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1))

        self.fc = [spectral_norm(nn.Linear(ndf * 8, 1))]
        self.fc = nn.Sequential(*self.fc)
        weights_init(self.fc)

        self.spatial_matrix = self.get_spatial_matrix()
        C, H, W = self.spatial_matrix.shape
        self.spatial_matrix = self.spatial_matrix.view(1, C, H, W)


        self.proj_conv = []
        self.proj_conv += [spectral_norm(nn.Conv2d(ndf, n, kernel_size=3, stride=1, padding=1))]

        self.proj_conv = nn.Sequential(*self.proj_conv)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.activation = self.leaky_relu

    def get_spatial_matrix(self, path="./model/prior_array.mat"):
        if not os.path.exists(path):
            raise FileExistsError("please put the spatial prior in .model/ \nThis prior comes from Yang Zou*, Zhiding Yu*, Vijayakumar Bhagavatula, Jinsong Wang. Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training. In ECCV'18 \nYou can download the prior at: https://www.dropbox.com/s/o6xac8r3z30huxs/prior_array.mat?dl=0")
        sprior = sio.loadmat(path)
        sprior = sprior["prior_array"]
        sprior = sprior[self.foreground_map]
        tensor_sprior = torch.tensor(sprior,
                                     dtype=torch.float64,
                                     device=torch.device('cuda:0')).float().cuda()
        return tensor_sprior

    def forward(self, input):

        x = self.conv1(input)
        x = self.activation(x)
        proj = self.proj_conv(x)
        proj_shape = proj.shape
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)

        # =================
        # project part
        # =================

        interp = nn.Upsample(size=proj_shape[-2:], align_corners=True, mode='bilinear')
        spatial_matrix = interp(self.spatial_matrix)
        input_foreground = input[:, self.foreground_map]
        input_foreground_resize = interp(input_foreground).detach()

        spatial_info = (1 + spatial_matrix) * input_foreground_resize
        proj = proj * spatial_info
        proj = torch.sum(proj, dim=(2, 3))

        # =================
        # block
        # =================
        x = self.conv4(x)
        x = self.activation(x)

        x = torch.sum(x, dim=(2, 3))
        x = self.fc(x)
        x = x + proj

        return x
