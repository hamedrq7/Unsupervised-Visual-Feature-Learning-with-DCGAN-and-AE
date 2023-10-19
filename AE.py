import torch 
import torch.nn as nn
import numpy as np 
from data_utils import get_cifar10_data, get_domain_net
from utils import *
import matplotlib.pyplot as plt
from AE_models import * 
from tqdm import trange, tqdm
# from pytorch_model_summary import summary
from typing import Dict, List, Tuple
import torchvision.transforms.functional as F
from torchsummary import summary

DATALOADER_SEED = 11

class AE_exp:
    def __init__(self, use_gpu: bool =True) -> None:
        self.use_gpu = use_gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")
        print(self.device)
        
        self.exp_name: str
        self.model: nn.Module
        self.criterion: nn.Module
        self.optim: torch.optim.Optimizer

        self.lr: float
        self.optim_str: str
        self.num_epochs: int
        self.batch_size: int
        self.model_name: str
        self.loss_fn_name: str
        
        self.path_to_save = 'exps/AEs'

        self.full_dataloaders: Dict[str, torch.utils.data.DataLoader]

    def env_setup(self, 

            dataset_name: str, model_name: str, loss_fn: str,
            batch_size = 128, image_size: int = 64, 
            dataloader_seed = 11):

        ### Data
        if dataset_name == 'cifar10':
            self.full_dataloaders, _, classes_str, _ = \
                get_cifar10_data(dataloader_seed, 
                    batch_size, 
                    shuffle=True
                    # , selected_classes=[1]
                    )
            self.num_classes = 10

        elif dataset_name == 'domainnet':
            self.group = 'mammals'
            self.image_size = image_size

            self.full_dataloaders, self.classes_str = \
                get_domain_net(self.group, normalize_data=False, \
                     image_size = image_size, batch_size=batch_size, \
                            pin_memory=True, num_workers=2, 
                            )
            
            self.num_classes = len(self.classes_str)                

        ### loss
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'bce':
            self.criterion = nn.BCELoss()
        
        ### model

        if model_name == 'Shallow_CNN_AE':
            if self.image_size == 32: 
                self.model = Shallow_CNN_AE()
            elif self.image_size == 64:
                self.model = Shallow_CNN_AE_64()
                

        self.model = self.model.to(self.device)
        print(summary(self.model, input_size=(3, self.image_size, self.image_size)))

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.loss_fn_name = loss_fn
        self.batch_size = batch_size

    def init_clf(self, lr = 0.01, clf_num_epohcs=5):
        z_dim = self.model.encoder.latent_dim[0] # 512
        self.clf = CLF(z_dim, 128, self.num_classes)
        self.clf = self.clf.to(self.device)
        self.clf_optim = torch.optim.Adam(self.clf.parameters(), lr = lr)
        self.clf_criterion = nn.CrossEntropyLoss() 


    def train_clf(self, loadedAE, lr = 0.01, clf_num_epochs: int = 10):
        # TODO: pass data through encode once, store and use it to train CLF
        
        z_dim = loadedAE.encoder.latent_dim[0] # 512 or 1024
        clf = CLF(z_dim, 128, self.num_classes)
        clf = clf.to(self.device)
        clf_optim = torch.optim.Adam(clf.parameters(), lr = lr)
        clf_criterion = nn.CrossEntropyLoss() 
        
        # self.model.eval()
        clf_acc_hist = {'train': [], 'test': []}


        # print(self.clf)

        for epoch in range(clf_num_epochs):

            for phase in ['train', 'test']:

                with torch.set_grad_enabled(phase == 'train'):
                    num_batches = len(self.full_dataloaders[phase])
                    # clf_running_loss = 0.0
                    running_corrects = 0
                    data_size = len(self.full_dataloaders[phase].dataset)

                    for batch_indx, (images, labels) in enumerate(self.full_dataloaders[phase]):
                        print(batch_indx)
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        with torch.no_grad():
                            latent_features = loadedAE.encoder(images)
                            # print(latent_features.shape)
                        
                        clf_output = clf(torch.flatten(latent_features, start_dim=1))
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
            
              
        
        custpm_plot_clf_acc(clf_acc_hist, 22, ['train', 'test'], f'CLF Acc hist at AE epoch {22}', f'{self.path_to_save}')

        return clf_acc_hist['train'][-1], clf_acc_hist['test'][-1] 

    def add_noise(self, inputs, noise_factor=0.3):
        noisy = inputs+torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy,0.,1.)
        return noisy

    def train(self, denoising: bool, learning_rate: float, optim_str: str, num_epochs: int = 3):
        
        self.lr = learning_rate
        self.optim_str = optim_str
        self.num_epochs = num_epochs
        
        # optim
        if optim_str == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.exp_name = f'Data_{self.dataset_name}-Denoise_{denoising}-Model_{self.model_name}-Loss_{self.loss_fn_name}-Epochs_{self.num_epochs}-{self.optim_str}-{self.lr}-{self.batch_size}'
        self.path_to_save += f'/{self.exp_name}'

        self.loss_hist = {'train': [], 'test': []}
        self.clf_acc_hist = {'train': [], 'test': []}


        for epoch in (pbar := tqdm(range(self.num_epochs), desc="trainL: {:.5f} - testL: {:.5f}".format(0.0, 0.0))):
            
            for phase in ['train', 'test']:
                data_size = len(self.full_dataloaders[phase].dataset)
                running_corrects = 0

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                num_batches = len(self.full_dataloaders[phase])
                running_loss = 0.0

                for batch_indx, (images, labels) in enumerate(self.full_dataloaders[phase]):
                    if denoising:
                        images = self.add_noise(images, 0.1)

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        
                        recons = self.model(images)
                        loss = self.criterion(recons, images)
                        running_loss += loss.item()

                        # update AE
                        if phase == 'train':
                            loss.backward()
                            self.optim.step()
                            self.optim.zero_grad()

                        if batch_indx == 0:
                            plot_recons_samples(recons=recons[0:64].cpu().detach().numpy().transpose(0, 2, 3, 1), 
                                                images=images[0:64].cpu().numpy().transpose(0, 2, 3, 1), 
                                                path_to_save=f'{self.path_to_save}/{self.exp_name}/images', 
                                                filename=f'{batch_indx}_{epoch} {phase}',
                                                title=f'E: {epoch} - phase: {phase} - Loss: {self.loss_fn_name} - indx: {batch_indx}')


                running_loss = running_loss / (num_batches)
                self.loss_hist[phase].append(running_loss)
                
                # progress bar
                if phase == 'train':
                    pbar.set_description_str("trainL: {:.5f} - testL: {:.5f}".format(running_loss, 0.0))
                else:
                    temp_l = self.loss_hist['train'][-1]
                    pbar.set_description_str("trainL: {:.5f} - testL: {:.5f}".format(temp_l, running_loss))

            self.save_model()

            # if epoch % 10 == 0 or epoch == num_epochs-1:
            #     clf_train_acc, clf_test_acc = self.train_clf(epoch, clf_num_epochs=5)
            #     self.clf_acc_hist['train'].append(clf_train_acc)
            #     self.clf_acc_hist['test'].append(clf_test_acc)

            # custom_plot_training_stats(self.clf_acc_hist, self.loss_hist, ['train', 'test'], title=f'{self.exp_name} Acc Loss', dir=f'{self.path_to_save}/{self.exp_name}')
            custom_plot_loss(self.loss_hist, ['train', 'test'], title=f'{self.exp_name} Acc Loss', dir=f'{self.path_to_save}/{self.exp_name}')


    def test(self, dataloader):
        
        self.criterion = nn.MSELoss()
        
        temp = {
            'features': [],
            'labels': [],
        }

        running_loss = 0.0
        num_batches = len(dataloader)
        # model is in self.device
        for batch_indx, (images, labels) in enumerate(dataloader): 
            images = images.to(self.device)

            with torch.no_grad():
                z = self.model.encoder(images)
                recon_images = self.model.decoder(z)

                loss = self.criterion(recon_images, images)
            
            running_loss += loss.item()
            temp['features'].append(z.cpu().detach().numpy())
            temp['labels'].append(labels.detach().numpy())
        
        print('avg loss: ', running_loss / num_batches)

        # save ? 

    def test_denoising(self, dataloader):
        
        self.criterion = nn.MSELoss()
        
        temp = {
            'features': [],
            'labels': [],
        }

        running_loss = 0.0
        num_batches = len(dataloader)
        # model is in self.device
        for batch_indx, (images, labels) in enumerate(dataloader): 
            images = images.to(self.device)

            with torch.no_grad():
                z = self.model.encoder(images)
                recon_images = self.model.decoder(z)

                loss = self.criterion(recon_images, images)
            
            running_loss += loss.item()
            temp['features'].append(z.cpu().detach().numpy())
            temp['labels'].append(labels.detach().numpy())
        
        print('avg loss: ', running_loss / num_batches)

        # save ? 

        

    def save_model(self):
        make_dir(self.path_to_save)
        torch.save(self.model, f'{self.path_to_save}/{self.model_name}.pt')

    def load_model(self, model_name: str): 
        self.model = torch.load(f'{self.path_to_save}/{model_name}')
        self.model = self.model.to(self.device)

######################################################### Phase1

###################
# Traning CLF at the end or at each epoch
###################
# exp = AE_exp()
# exp.env_setup(data_name='cifar10', model_name='Shallow_CNN_AE',  loss_fn='mse')
# exp.train(False, 0.01, 'adam', 10)
# exp.save_model()


###################
# Denoising vs normal AE
###################
# exp = AE_exp()
# exp.env_setup(data_name='cifar10', model_name='Shallow_CNN_AE',  loss_fn='mse')
# exp.train(False, 0.01, 'adam', 50)
# exp.save_model()

# exp = AE_exp()
# exp.env_setup(data_name='cifar10', model_name='Shallow_CNN_AE',  loss_fn='mse')
# exp.train(False, 0.01, 'adam', 50)
# exp.save_model()

###################
# normal AE phase 1
###################
# exp = AE_exp()
# exp.env_setup(data_name='cifar10', model_name='Shallow_CNN_AE',  
#     loss_fn='mse', batch_size=64)
# exp.train(False, 0.01, 'adam', 60)
# exp.save_model()


################################################# Phase 2
# normal AE in domain net
# exp = AE_exp()
# exp.env_setup(dataset_name='domainnet', model_name='Shallow_CNN_AE',  
#     loss_fn='mse', batch_size=64, image_size=64)
# exp.train(False, 0.01, 'adam', 50)



#### ACC
exp = AE_exp()
exp.env_setup(dataset_name='domainnet', model_name='Shallow_CNN_AE',  
    loss_fn='mse', batch_size=64, image_size=64)

path_to_model = '/content/drive/MyDrive/DomainNet/Shallow_CNN_AE.pt'
AE = torch.load(path_to_model)
exp.train_clf(AE, clf_num_epochs=5)