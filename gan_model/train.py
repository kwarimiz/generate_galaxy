#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import time

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets 
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from my_dataset import MyDataset

from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 设置一个随机种子，方便进行可重复性实验
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = 'your data path'

batch_size = 256
lr = 0.00003
num_epochs = 5000

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

num_gpu = 4

# Decide which device we want to run on
device = torch.device("cuda:0")


# In[15]:

dataset = MyDataset(dataroot,
                           transform=transforms.Compose([
                               transforms.CenterCrop(180),
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=8)


# Create the generator
netG = Generator(num_gpu).to(device)
netG = nn.DataParallel(netG, list(range(num_gpu)))

if os.path.exists('./pth/G_best.pth'):
    netG.load_state_dict(torch.load('./pth/G_best.pth'))
else:
    netG.apply(weights_init)


#Create thr discriminator
netD = Discriminator(num_gpu).to(device)
netD = nn.DataParallel(netD, list(range(num_gpu)))

if os.path.exists('./pth/D_best.pth'):
    netD.load_state_dict(torch.load('./pth/D_best.pth'))
else:
    netD.apply(weights_init)
print('net load done')

# Initialize BCELoss function
criterion = nn.BCELoss()


# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

G_best_loss_path = './pth/G_best.pth'
D_best_loss_path = './pth/D_best.pth'
G_latest_loss_path = './pth/G_latest.pth'
D_latest_loss_path = './pth/D_latest.pth'

G_best_loss = 100.00                     # init G and D loss,to get best loss
D_best_loss = 100.00

# For each epoch
for epoch in range(num_epochs):
    import time
    start = time.time()
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 20 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):

            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()

            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            i = vutils.make_grid(fake, padding=2, normalize=True)
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(np.transpose(i, (1, 2, 0)))
            plt.axis('off')  # 关闭坐标轴
            plt.savefig("out/%d_%d.png" % (epoch, iters))
            plt.close(fig)
        iters += 1
    print('time:', time.time() - start)

    if errG.item()< G_best_loss:
        G_best_loss = errG.item()
        torch.save(netG.state_dict(),G_best_loss_path)

    if errD.item()< D_best_loss:
        D_best_loss = errD.item()
        torch.save(netD.state_dict(),D_best_loss_path)

    torch.save(netG.state_dict(),G_latest_loss_path)
    torch.save(netD.state_dict(),D_latest_loss_path)
