import torch
import torch.nn as nn
from torchvision import models


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator_64(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1):
        super(Generator_64, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

class Generator_128(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1):
        super(Generator_128, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu

        coeffs = [8, 6, 4, 2, 1]
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * coeffs[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * coeffs[0]),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * coeffs[0], ngf * coeffs[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * coeffs[1]),
            nn.ReLU(True),
            # state size. ``(ngf*6) x 8 x 8``
            nn.ConvTranspose2d(ngf * coeffs[1], ngf * coeffs[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * coeffs[2]),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 16 x 16``
            nn.ConvTranspose2d(ngf * coeffs[2], ngf * coeffs[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * coeffs[3]),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 32 x 32``
            nn.ConvTranspose2d(ngf * coeffs[3], ngf * coeffs[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * coeffs[4]),
            nn.ReLU(True),
            # state size. ``(ngf*1) x 64 x 64``
            nn.ConvTranspose2d(ngf*coeffs[4], nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. ``(nc) x 128 x 128``
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=1):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.Conv2d(ngf, nc, 1, 1, 0),
            nn.Tanh()
            # state size. ``(3) x 32 x 32``
        )

    def forward(self, input):
        return self.main(input)


class Discriminator_64(nn.Module):
    def __init__(self, nc, ndf, ngpu=1):
        super(Discriminator_64, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 4 x 4``
        )
        self.latent_dim = (ndf * 2, 4, 4)
        self.clf_layer = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 x 1 x 1 
        )

    def forward(self, input):
        features =  self.main(input)

        return features, self.clf_layer(features)

class Discriminator_128(nn.Module):
    def __init__(self, nc, ndf, ngpu=1):
        super(Discriminator_128, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        coeffs = [1, 2, 4, 6, 8]
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 128``
            
            nn.Conv2d(nc, ndf*coeffs[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 64 x 64``

            nn.Conv2d(ndf*coeffs[0], ndf*coeffs[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*coeffs[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 32 x 32``
            
            nn.Conv2d(ndf*coeffs[1], ndf*coeffs[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*coeffs[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 16 x 16``

            nn.Conv2d(ndf*coeffs[2], ndf*coeffs[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*coeffs[3]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*6) x 8 x 8``

            nn.Conv2d(ndf*coeffs[3], ndf*coeffs[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*coeffs[4]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            
            nn.Conv2d(ndf*coeffs[4], 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        
class Discriminator(nn.Module):
    def __init__(self, nc, ndf, ngpu=1):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 32 x 32``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 16 x 16``
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 8 x 8``
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 4 x 4``
        )
        self.latent_dim = (ndf * 4, 4, 4)

        self.clf_layer = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. ``1 x 1 x 1``
        )

    def forward(self, input):
        features =  self.main(input)

        return features, self.clf_layer(features)
    
class CLF(nn.Module):
    def __init__(self, input_size, hidden_dim, out_dim) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # print('x shape', x.shape)

        x = self.fc1(x)
        # ? 
        x = nn.functional.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        
        return x 
