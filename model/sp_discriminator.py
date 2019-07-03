import torch.nn as nn
from torch.nn.utils import spectral_norm


class SP_FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(SP_FCDiscriminator, self).__init__()

		self.conv1 = spectral_norm(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1))
		self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1))
		self.conv3 = spectral_norm(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1))
		self.conv4 = spectral_norm(nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1))
		self.classifier = spectral_norm(nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1))

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.activation = self.leaky_relu


	def forward(self, x):
		x = self.conv1(x)
		x = self.activation(x)
		x = self.conv2(x)
		x = self.activation(x)
		x = self.conv3(x)
		x = self.activation(x)
		x = self.conv4(x)
		x = self.activation(x)
		x = self.classifier(x)

		return x
