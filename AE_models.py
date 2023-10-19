import torch
import torch.nn as nn
from torchvision import models

# vgg16 = models.vgg16_bn(pretrained=False)
# print(vgg16)

class Shallow_Encoder_64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
      
        # channels = [16, 32, 64, 128, 256]
        channels = [16, 32, 64, 128, 256] # , 512
        
        filter_sizes = [3, 3, 3, 3, 3]
        stride_sizes = [1, 1, 1, 1, 1]
        padding_sizes = [1, 1, 1, 1, 1]

        # [3, 64, 64]
        bias_conv=False
        self.block0 = nn.Sequential(
            nn.Conv2d(3, channels[0], filter_sizes[0], stride_sizes[0], padding_sizes[0], bias=bias_conv),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),

            nn.Conv2d(channels[0], channels[0], filter_sizes[0], stride_sizes[0], padding_sizes[0], bias=bias_conv),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        )
        self.pool0 = nn.AvgPool2d(2, 2)
        
        # [channels_0, 32, 32] 16
        self.block1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], filter_sizes[1], stride_sizes[1], padding_sizes[1], bias=bias_conv),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),

            nn.Conv2d(channels[1], channels[1], filter_sizes[1], stride_sizes[1], padding_sizes[1], bias=bias_conv),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
        )
        self.pool1 = nn.AvgPool2d(2, 2)
        
        # [channels_1, 16, 16] 32
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], filter_sizes[2], stride_sizes[2], padding_sizes[2], bias=bias_conv),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),

            nn.Conv2d(channels[2], channels[2], filter_sizes[2], stride_sizes[2], padding_sizes[2], bias=bias_conv),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
        )
        self.pool2 = nn.AvgPool2d(2, 2)
        
        # [channels_2, 8, 8] 64
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], filter_sizes[3], stride_sizes[3], padding_sizes[3], bias=bias_conv),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),

            nn.Conv2d(channels[3], channels[3], filter_sizes[3], stride_sizes[3], padding_sizes[3], bias=bias_conv),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
        ) 
        self.pool3 = nn.AvgPool2d(2, 2)

        # [channels_3, 4, 4] 128
        self.block4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], filter_sizes[4], stride_sizes[4], padding_sizes[4], bias=bias_conv),
            nn.BatchNorm2d(channels[4]),
            nn.PReLU()
        ) # (channels_4, 4, 4) 256
        
        # self.pool4 = nn.AvgPool2d(2, 2)

        # (channels_4, 4, 4) 256
        self.block5 = nn.Sequential(
            nn.Conv2d(channels[4], 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.Tanh()
        ) # (512, 2, 2)
        
        self.pool5 = nn.AdaptiveAvgPool2d(1)
        
        self.latent_dim = (512, 1, 1)

    def forward(self, x):
        # x.shape -> [3, 64, 64]
        x = self.pool0(self.block0(x)) 
        x = self.pool1(self.block1(x)) 
        x = self.pool2(self.block2(x)) 
        x = self.pool3(self.block3(x))
        x = self.block4(x)

        z = self.pool5(self.block5(x))

        return z

class Shallow_Decoder_64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_c = 512
        out_c = [256, 128, 64, 32, 16, 8]

        # 512, 1, 1

        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c[0], kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_c[0]),
            nn.PReLU(), # ? 
        ) # 256, 3, 3

        
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(out_c[0], out_c[1], kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU(),
        ) # 128, 4, 4

        
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(out_c[1], out_c[2], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU(),
        ) # 64, 8, 8
        
        
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(out_c[2], out_c[3], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[3]),
            nn.ReLU(),
        ) # 32, 16, 16

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(out_c[3], out_c[4], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[4]),
            nn.ReLU(), 
        ) # 16, 32, 32

        
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(out_c[4], out_c[5], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[5]),
            nn.Tanh(), # ? 
        ) # 8, 64, 64

        self.block6 = nn.Sequential( 
            nn.Conv2d(out_c[5], 3, kernel_size=(1, 1), stride=1, padding=0, bias=True),
            nn.Sigmoid() 
        ) # 3, 64, 64

    def forward(self, z):
        # z -> [512, 1, 1]
        x = self.block0(z)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x_hat = self.block6(x)

        return x_hat

class Shallow_CNN_AE_64(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Shallow_Encoder_64()
        self.decoder = Shallow_Decoder_64()

    def forward(self, x):
        latent_features = self.encoder(x)
        recon = self.decoder(latent_features)

        return recon
    


# Change Pooling layers to max layer (except for last one)
class Shallow_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
      
        # channels = [16, 32, 64, 128, 256]
        channels = [32, 64, 128, 256]
        
        filter_sizes = [3, 3, 3, 3, 3]
        stride_sizes = [1, 1, 1, 1, 1]
        padding_sizes = [1, 1, 1, 1, 1]

        # [3, 32, 32]
        bias_conv=False
        self.block0 = nn.Sequential(
            nn.Conv2d(3, channels[0], filter_sizes[0], stride_sizes[0], padding_sizes[0], bias=bias_conv),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),

            nn.Conv2d(channels[0], channels[0], filter_sizes[0], stride_sizes[0], padding_sizes[0], bias=bias_conv),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        )
        self.pool0 = nn.AvgPool2d(2, 2)
        
        # [channels_0, 16, 16]
        self.block1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], filter_sizes[1], stride_sizes[1], padding_sizes[1], bias=bias_conv),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),

            nn.Conv2d(channels[1], channels[1], filter_sizes[1], stride_sizes[1], padding_sizes[1], bias=bias_conv),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
        )
        self.pool1 = nn.AvgPool2d(2, 2)
        
        # [channels_1, 8, 8]
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], filter_sizes[2], stride_sizes[2], padding_sizes[2], bias=bias_conv),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),

            nn.Conv2d(channels[2], channels[2], filter_sizes[2], stride_sizes[2], padding_sizes[2], bias=bias_conv),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
        )
        self.pool2 = nn.AvgPool2d(2, 2)
        
        # [channels_2, 4, 4]
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], filter_sizes[3], stride_sizes[3], padding_sizes[3], bias=bias_conv),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),

            nn.Conv2d(channels[3], channels[3], filter_sizes[3], stride_sizes[3], padding_sizes[3], bias=bias_conv),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
        ) # [channels_3, 4, 4]

        ### use s=1, p=0, f=3 to make the feature maps 2x2 then avg pool
        self.block4 = nn.Sequential(
            nn.Conv2d(channels[3], 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.Tanh()
        ) # (512, 2, 2)
        
        self.pool4 = nn.AvgPool2d(2, 2)
        
        self.latent_dim = (512, 1, 1)

        ### use 1x1 conv to reduce channels
        # z = 512
        # z_channel = 128 # 512 // (4 * 4) # 32
        # self.one_one_conv = nn.Sequential(
        #     nn.Conv2d(channels[3], z_channel, 1),
        #     nn.BatchNorm2d(z_channel),
        #     # nn.Tanh(), # Change ? 
        # )

        # self.pool3 = nn.AdaptiveAvgPool2d((1, 1)) # no spatial         

    def forward(self, x):
        # x.shape -> [3, 32, 32]
        x = self.pool0(self.block0(x)) # [c0, 16, 16]
        x = self.pool1(self.block1(x)) # [c1, 8, 8]
        x = self.pool2(self.block2(x)) # [c2, 4, 4]
        x = self.block3(x) # [c3, 4, 4]
        
        # x = self.pool3(x) # [c3, 1, 1] # no spatial
        # x = self.one_one_conv(x) # 1x1 conv to reduce #channels
        
        x = self.block4(x)
        z = self.pool4(x)

        return z

class Shallow_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_c = 512
        out_c = [256, 128, 64, 32, 16]

        # 512, 1, 1
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c[0], kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_c[0]),
            nn.Tanh(), # ? 
        )

        # 256, 3, 3
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(out_c[0], out_c[1], kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU(),
        )

        # 128, 4, 4
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(out_c[1], out_c[2], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU(),
        )
        
        # 64, 8, 8
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(out_c[2], out_c[3], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[3]),
            nn.ReLU(),
        )

        # 32, 16, 16
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(out_c[3], out_c[4], kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c[4]),
            nn.Tanh(), # ? 
        )

        # 16, 32, 32
        self.block5 = nn.Sequential( 
            nn.Conv2d(out_c[4], 3, kernel_size=(1, 1), stride=1, padding=0, bias=True),
            nn.Sigmoid() 
        )
    

    def forward(self, z):
        # z -> [512, 1, 1]
        x = self.block0(z)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x_hat = self.block5(x)

        return x_hat

class Shallow_CNN_AE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Shallow_Encoder()
        self.decoder = Shallow_Decoder()

    def forward(self, x):
        latent_features = self.encoder(x)
        recon = self.decoder(latent_features)

        return latent_features, recon
    


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
