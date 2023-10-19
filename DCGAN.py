import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from data_utils import *
from utils import *
from typing import List, Dict, Tuple
from GAN_models import * 
from tqdm import trange, tqdm
from torchsummary import summary

class GAN_exp():
    def __init__(self, use_gpu = True) -> None:
        # Set random seed for reproducibility
        manualSeed = 999
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        # torch.use_deterministic_algorithms(True) # Needed for reproducible results

        
        self.use_gpu = use_gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")
        print(self.device)

        self.exp_name: str

        self.gen: nn.Module
        self.disc: nn.Module

        self.criterion: nn.Module
        self.criterion_g: nn.Module
        self.criterion_d: nn.Module
        
        self.optim: torch.optim.Optimizer

        self.lr: float
        self.optim_str: str
        self.num_epochs: int
        self.batch_size: int
        self.gen_name: str
        self.disc_name: str
        
        self.path_to_save = 'exps/GANs'


        # **
        self.dataset_name: str # cifar10 or domainnet
        
        self.full_dataloaders: Dict[str, torch.utils.data.DataLoader]
        
    def env_setup(self, model_name: str,
                    dataset_name: str, 
                    group: str, 
                    image_size: int,
                    normal_data: bool, 
                    nc=3, nz=100, ngf=64, ndf=64,
                    batch_size = 128, dataloader_seed = 11,
                    return_test_data: bool = False):

        self.batch_size = batch_size
        self.image_size = image_size
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.dataset_name = dataset_name
        self.group = group # mammals, road transporations
        self.normal_data = normal_data
        # ? 
        self.model_name = model_name

        # Create the dataloader
        if dataset_name == 'cifar10':
            self.full_dataloaders, _, classes_str, _ = \
                        get_cifar10_data(11,
                            self.batch_size,
                            shuffle=True,
                            # , selected_classes=[1]
                            )
            self.gen = Generator(nz, ngf, nc).to(self.device)
            self.disc = Discriminator(nc, ndf).to(self.device)
            self.num_classes = 10

        elif dataset_name == 'domainnet':
            self.full_dataloaders, self.classes_str = \
                get_domain_net(group, normalize_data=normal_data, \
                            image_size = image_size, batch_size=batch_size,
                             pin_memory=True, num_workers=2,
                            return_test_data=return_test_data)
            
            self.num_classes = len(self.classes_str)

            if image_size == 32:
                self.gen = Generator(nz, ngf, nc).to(self.device)
                self.disc = Discriminator(nc, ndf).to(self.device)
            elif image_size == 64:
                self.gen = Generator_64(nz, ngf, nc).to(self.device)
                self.disc = Discriminator_64(nc, ndf).to(self.device)
            elif image_size == 128:
                self.gen = Generator_128(nz, ngf, nc).to(self.device)
                self.disc = Discriminator_128(nc, ndf).to(self.device)
            else:
                print(f'no model for the image size={image_size}')


        # print('Generator: ', self.gen)
        # print('Discriminator: ', self.disc)
        summary(self.gen, (100, 1, 1))
        summary(self.disc, (3, image_size, image_size))
        # add torch summary

        # apply weight init: 
        self.gen.apply(weights_init)
        self.disc.apply(weights_init)

        self.criterion = nn.BCELoss()

    def train_clf(self, loaded_disc, lr = 0.01, clf_num_epochs: int = 8):
        # TODO: pass data through encode once, store and use it to train CLF
        
        z_dim = loaded_disc.latent_dim[0] * loaded_disc.latent_dim[1] * loaded_disc.latent_dim[2] # 

        self.clf = CLF(z_dim, 128, self.num_classes)

        self.clf = self.clf.to(self.device)
        clf_optim = torch.optim.Adam(self.clf.parameters(), lr = lr)
        clf_criterion = nn.CrossEntropyLoss() 
        
        loaded_disc.eval()
        clf_acc_hist = {'train': [], 'test': []}

        print(self.clf)

        for epoch in range(clf_num_epochs):

            for phase in ['train', 'test']:

                with torch.set_grad_enabled(phase == 'train'):
                    num_batches = len(self.full_dataloaders[phase])
                    # clf_running_loss = 0.0
                    running_corrects = 0
                    data_size = len(self.full_dataloaders[phase].dataset)

                    for batch_indx, (images, labels) in enumerate(self.full_dataloaders[phase]):
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        with torch.no_grad():
                            latent_features, _ = loaded_disc(images)
                            # print(latent_features.shape)
                        
                        clf_output = self.clf(torch.flatten(latent_features, start_dim=1))
                        _, clf_preds = torch.max(clf_output, dim=1)
                        clf_loss = clf_criterion(clf_output, labels)

                        # clf_running_loss += clf_loss.item()

                        if phase == 'train':
                            clf_loss.backward()
                            clf_optim.step()
                            clf_optim.zero_grad()

                        running_corrects += torch.sum(clf_preds == labels)

                    # clf_running_loss = clf_running_loss / (num_batches)
                    clf_acc_hist[phase].append((running_corrects.double() / data_size).cpu().numpy())
                
        
        custpm_plot_clf_acc(clf_acc_hist, 80, ['train', 'test'], f'CLF Acc hist at GAN epoch {80}', f'{self.path_to_save}')
        # self.disc.train()

        return clf_acc_hist['train'][-1], clf_acc_hist['test'][-1] 


    def train(self, num_epochs: int, 
              learning_rate: float=0.0002, beta1=0.5
              ):
        self.lr = learning_rate
        self.num_epochs = num_epochs

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, self.gen.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.disc_optim = optim.Adam(self.disc.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        self.gen_optim = optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta1, 0.999))

        self.exp_name = f'imsize_{self.image_size}-normalImg_{self.normal_data}-ngf_{self.ngf}-ndf_{self.ndf}-lr_{self.lr}-b1_{beta1}-domain_{self.dataset_name}- /group_{self.group}-bs_{self.batch_size}-num_epochs_{self.num_epochs}'
        print(self.exp_name)

        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []

        self.avg_G_loss_hist = []
        self.avg_D_loss_hist = []
        self.avg_Dx_hist = []
        self.avg_D_Gz_hist = []

        self.clf_acc_hist = {'train': [], 'test': []}
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in trange(self.num_epochs):
            
            running_G_loss = 0.0
            running_D_loss = 0.0
            running_Dx = 0.0
            running_D_Gz = 0.0
            num_batches = len(self.full_dataloaders['train'])

            for i, (images, class_labels) in enumerate(self.full_dataloaders['train']):
                if epoch == 0:
                    print(i, end='\t')
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.disc.zero_grad()
                # Format batch
                real_cpu = images.to(self.device)
                b_size = real_cpu.size(0) # batch size 
        
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                _, output = self.disc(real_cpu)
                output = output.view(-1)
                
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item() # best case this is 1

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.gen.nz, 1, 1, device=self.device) # (batch_size, 100, 1, 1)
                # Generate fake image batch with G
                fake = self.gen(noise)
                label.fill_(fake_label)
        
                # Classify all fake batch with D
                _, output = self.disc(fake.detach()) # why call detach? 
                output = output.view(-1)

                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated 
                # (summed) with previous gradients
                errD_fake.backward()
        
                D_G_z1 = output.mean().item() # best case 0
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.disc_optim.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.gen.zero_grad() # grads have been accumulated because of forward pass in training disc
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                _, output = self.disc(fake)
                output = output.view(-1)
                
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.gen_optim.step()

                # Output training stats
                # if i % 5 == 0:
                #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                #         % (epoch, num_epochs, i, len(self.full_dataloaders['train']),
                #             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                running_D_loss += errD.item() 
                running_G_loss += errG.item()
                running_Dx += D_x
                running_D_Gz += D_G_z1

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 50 == 0) or ((epoch == num_epochs-1) and (i == len(self.full_dataloaders['train'])-1)):
                    with torch.no_grad():
                        fake = self.gen(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

            # monitor
            self.avg_G_loss_hist.append(running_G_loss / num_batches)
            self.avg_D_loss_hist.append(running_D_loss / num_batches)
            self.avg_Dx_hist.append(running_Dx / num_batches)
            self.avg_D_Gz_hist.append(running_D_Gz / num_batches)
            
            ### Pass to CLF 
            # if epoch % 10 == 0 or epoch == num_epochs-1:
            #     clf_train_acc, clf_test_acc = self.train_clf(epoch, clf_num_epochs=5)
            #     self.clf_acc_hist['train'].append(clf_train_acc)
            #     self.clf_acc_hist['test'].append(clf_test_acc)

            # plot images
            self.plot_stats(epoch, G_losses, D_losses, self.path_to_save, img_list)
            
            torch.save(self.gen, f'{self.path_to_save}/{self.exp_name}/gen.pt')
            torch.save(self.disc, f'{self.path_to_save}/{self.exp_name}/disc.pt')
            # torch.save(self.clf, f'{self.path_to_save}/{self.exp_name}/clf.pt')
            
            cutsom_plot_GAN(self.avg_G_loss_hist, self.avg_D_Gz_hist, self.avg_D_loss_hist, self.avg_Dx_hist, title=f'Gan Stats {self.exp_name}', dir=f'{self.path_to_save}/{self.exp_name}')

        """
        Finally, we will do some statistic reporting and at the end of each epoch we will push our fixed_noise batch through the generator to visually track the progress of G’s training. The training statistics reported are:

        Loss_D - discriminator loss calculated as the sum of losses for the all real and all fake batches (log(D(x))+log(1−D(G(z)))log(D(x))+log(1−D(G(z)))).

        Loss_G - generator loss calculated as log(D(G(z)))log(D(G(z)))

        D(x) - the average output (across the batch) of the discriminator for the all real batch. This should start close to 1 then theoretically converge to 0.5 when G gets better. Think about why this is.

        D(G(z)) - average discriminator outputs for the all fake batch. The first number is before D is updated and the second number is after D is updated. These numbers should start near 0 and converge to 0.5 as G gets better. Think about why this is.

        """
        # custpm_plot_clf_acc(self.clf_acc_hist, -1, ['train', 'test'], f'CLF Acc hist', f'{self.path_to_save}/{self.exp_name}')


    def plot_stats(self, epoch, G_losses, D_losses, path_to_save, img_list):
        make_dir(f'{path_to_save}/{self.exp_name}')
        
        if epoch == self.num_epochs-1:
            plt.figure(figsize=(10,5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses,label="G")
            plt.plot(D_losses,label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'{path_to_save}/{self.exp_name}/GANs Loss-{epoch}')
            plt.clf()

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(self.full_dataloaders['train']))

        # Plot the real images
        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
        
        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.savefig(f'{path_to_save}/{self.exp_name}/real vs fake images-{epoch}')


################################### Phase 1

################
# Phase1, DCGAN images + clf acc cifar10
################
# batch_size = 64
# num_epochs = 2

# exp = GAN_exp()
# exp.env_setup('default', dataset_name='cifar10',
#     group='', image_size=32, normal_data=True,
#     ngf = 64, ndf = 64, batch_size=batch_size)
# exp.train(num_epochs=num_epochs, learning_rate=0.0002, beta1=0.7)



################################### Phase 2

################
# Phase2, DCGAN images + clf acc Domainnet
################
# for image_size in [64]: # , 128, 224
#     for ndf in [64]:
#         for ngf in [64]: 
#             for lr in [0.0002]: 
#                 for normal_data in [True]:
#                     dataset_name = 'domainnet'
#                     group = 'mammals'
#                     batch_size = 64
#                     num_epochs = 80
                    
#                     exp = GAN_exp()
#                     exp.env_setup('default', dataset_name=dataset_name,
#                         group=group, image_size=image_size, normal_data=normal_data,
#                         ngf = ngf, ndf = ndf, batch_size=batch_size)
#                     exp.train(num_epochs=num_epochs, learning_rate=lr, 
#                         beta1=0.7)


### ACC
path_to_model = '/content/drive/MyDrive/exps/GANs/imsize_64-normalImg_True-ngf_64-ndf_64-lr_0.0002-b1_0.7-domain_domainnet- /group_mammals-bs_64-num_epochs_80/disc.pt'
disc = torch.load(path_to_model)

for image_size in [64]: # , 128, 224
    for ndf in [64]:
        for ngf in [64]: 
            for lr in [0.0002]: 
                for normal_data in [True]:
                    dataset_name = 'domainnet'
                    group = 'mammals'
                    batch_size = 64
                    num_epochs = 80
                    
                    exp = GAN_exp()
                    exp.env_setup('default', dataset_name=dataset_name,
                        group=group, image_size=image_size, normal_data=normal_data,
                        ngf = ngf, ndf = ndf, batch_size=batch_size, return_test_data=True)
                    exp.train_clf(disc)
