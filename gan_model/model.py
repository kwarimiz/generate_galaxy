import torch.nn as nn

num_chanels = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
num_G_features = 64

# Size of feature maps in discriminator
num_D_features = 64

# 权重初始化函数，为生成器和判别器模型初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, num_gpu):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, num_G_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_G_features * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_G_features * 8, num_G_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_G_features * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( num_G_features * 4, num_G_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_G_features * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( num_G_features * 2, num_G_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_G_features),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( num_G_features, num_chanels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_gpu):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_chanels, num_D_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_D_features, num_D_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_D_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_D_features * 2, num_D_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_D_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_D_features * 4, num_D_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_D_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_D_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
