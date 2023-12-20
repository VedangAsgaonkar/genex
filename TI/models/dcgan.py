import torch
import torchvision
import torch.nn as nn

class DCGANDecoder(nn.Module):
    def __init__(self):
        super(DCGANDecoder, self).__init__()
        nz = 100
        ngf = 128
        nc = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Unflatten(1,(nz,1,1)),
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32*32
            
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64*64
        )

    def forward(self, input):
        factor = 3
        return factor*self.main(input)
    
    

# class DCGANDecoder(nn.Module):
#     def __init__(self):
#         super(DCGANDecoder, self).__init__()
#         nz = 100
#         ngf = 128
#         nc = 3
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.Unflatten(1,(nz,1,1)),
#             nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 32 x 32
#         )

#     def forward(self, input):
#         factor = 2
#         return factor*self.main(input)