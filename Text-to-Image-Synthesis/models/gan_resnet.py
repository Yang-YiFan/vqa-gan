import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed
import pdb
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3


class generator(nn.Module):
	def __init__(self):
		super(generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.latent_dim = self.noise_dim + self.projected_embed_dim
		self.ngf = 64

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)

		# based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
		self.netG = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			#BasicBlock(self.ngf * 8, self.ngf * 8),
			BasicBlock(self.ngf * 8, self.ngf * 4, downsample=conv3x3(self.ngf * 8, self.ngf * 4)),
			nn.Upsample(scale_factor=2),
			# state size. (ngf*4) x 8 x 8
			#BasicBlock(self.ngf * 4, self.ngf * 4),
			BasicBlock(self.ngf * 4, self.ngf * 2, downsample=conv3x3(self.ngf * 4, self.ngf * 2)),
			nn.Upsample(scale_factor=2),
			# state size. (ngf*2) x 16 x 16
			#BasicBlock(self.ngf * 2, self.ngf * 2),
			BasicBlock(self.ngf * 2, self.ngf, downsample=conv3x3(self.ngf * 2, self.ngf)),
			nn.Upsample(scale_factor=2),
			# state size. (ngf) x 32 x 32
			BasicBlock(self.ngf, self.ngf),
			nn.Upsample(scale_factor=2),
			# state size. (ngf) x 64 x 64
			BasicBlock(self.ngf, self.ngf),
			BasicBlock(self.ngf, self.ngf),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			nn.Conv2d(self.ngf, 3, [3, 3], padding=1),
			nn.Tanh()
			 # state size. (num_channels) x 64 x 64
			)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, embed_vector, z):

		projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_embed, z], 1)
		output = self.netG(latent_vector)

		return output

def make_downsample(inplanes, planes, stride=2):
	downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes)
				)
	return downsample

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16

		self.netD_1 = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(self.num_channels, self.ndf, 3, stride = 2, padding = 1, bias=False),
			nn.BatchNorm2d(self.ndf),
			nn.ReLU(True),
			# state size. (ndf) x 32 x 32
			BasicBlock(self.ndf, self.ndf * 2, stride = 2, downsample=make_downsample(self.ndf, self.ndf * 2)),
			BasicBlock(self.ndf * 2, self.ndf * 2),
			# state size. (ndf*2) x 16 x 16
			BasicBlock(self.ndf * 2, self.ndf * 4, stride = 2, downsample=make_downsample(self.ndf * 2, self.ndf * 4)),
			BasicBlock(self.ndf * 4, self.ndf * 4),
			# state size. (ndf*4) x 8 x 8
			BasicBlock(self.ndf * 4, self.ndf * 8, stride = 2, downsample=make_downsample(self.ndf * 4, self.ndf * 8)),
			BasicBlock(self.ndf * 8, self.ndf * 8),
			# state size. (ndf*8) x 4 x 4
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)	

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, inp, embed):
		x_intermediate = self.netD_1(inp)
		x = self.projector(x_intermediate, embed)
		x = self.netD_2(x)

		return x.view(-1, 1).squeeze(1) , x_intermediate

'''
class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 1024
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16

		self.netD_1 = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
		)

		self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)	

	def forward(self, inp, embed):
		x_intermediate = self.netD_1(inp)
		x = self.projector(x_intermediate, embed)
		x = self.netD_2(x)

		return x.view(-1, 1).squeeze(1) , x_intermediate
'''