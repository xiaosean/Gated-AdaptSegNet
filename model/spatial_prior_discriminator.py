import torch as torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Spatial_Prior_Discriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(Spatial_Prior_Discriminator, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))

        # ==================== #
        #    model pre         #
        # ==================== #
        self.model_pre = []
        # # channe = 64
        self.model_pre += [spectral_norm(nn.Conv2d(num_classes, ndf, 4, 2, 1))]
        self.model_pre += [nn.LeakyReLU(0.2)]
        # # # channe = 128
        self.model_pre += [spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))]
        self.model_pre += [nn.LeakyReLU(0.2)]
        # # # channe = 256
        self.model_pre += [spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1))]
        self.model_pre += [nn.LeakyReLU(0.2)]

        # use cGANs with projection
        # ==================== #
        #   proj conv          #
        # ==================== #
        self.proj_conv = []
        self.proj_conv += [spectral_norm(nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=1, padding=1))]
        self.model_block = []

        # channel = 512
        self.model_block += [spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1))]
        self.model_block += [nn.LeakyReLU(0.2)]

        self.model_block_out_size = ndf*8
        # ==================== #
        #          fc          #
        # ==================== #
        # self.fc = nn.Linear(ndf * 8, 1)
        self.fc = spectral_norm(nn.Linear(self.model_block_out_size, 1))

        # ==================== #
        #     model_classifier #
        # ==================== #
        # create model
        self.model_pre = nn.Sequential(*self.model_pre)
        self.model_block = nn.Sequential(*self.model_block)
        self.proj_conv = nn.Sequential(*self.proj_conv)

    def forward(self, x, label=None):
        assert label is not None, "plz give me label let me train discriminator"
        x = self.model_pre(x)

        proj_x = self.proj_conv(x)

        x = self.model_block(x)
        # global sum pooling
        x = torch.sum(x, dim=(2, 3))

        x = x.view(-1, self.model_block_out_size)
        output = self.fc(x)
        output += self.gamma*torch.sum(proj_x*label)

        return output, proj_x


