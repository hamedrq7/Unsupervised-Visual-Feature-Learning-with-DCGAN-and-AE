import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import random 
import matplotlib.pyplot as plt
from utils import *
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageFolder


def get_cifar10_data(data_loader_seed, batch_size=128, 
        shuffle: bool = True, selected_classes=list(np.arange(10)),
        normalize_data: bool = False): 
    
    # Dont normalize data for AEs

    if normalize_data:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            ])

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)

    g = torch.Generator()
    g.manual_seed(data_loader_seed)

    

    trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                        download=True, transform=transform)

    train_idxs = torch.where(torch.isin(torch.asarray(trainset.targets), torch.asarray(selected_classes)))[0]
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            # sampler=SubsetRandomSampler(train_idxs, g),
                                            shuffle=shuffle)


    testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                        download=True, transform=transform)
    test_idxs = torch.where(torch.isin(torch.asarray(testset.targets), torch.asarray(selected_classes)))[0]
    testloader = DataLoader(testset, batch_size=batch_size,
                                            worker_init_fn=seed_worker,
                                            generator=g,  
                                            # sampler=SubsetRandomSampler(test_idxs),
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = 10
    input_size = (3 ,32 ,32)

    full_data_loaders = {
        'train': trainloader,
        'test':  testloader,
    }

    return full_data_loaders, input_size, classes, batch_size

def get_samples(dataloader, num_samples: int, images_are_normalized: bool) -> np.ndarray: 
    samples = [
        [], [], [], [], [], [], [], [], [], []
    ]
    
    for batch_indx, (images, labels) in enumerate(dataloader): 
        # images are normalized using mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5),
        # so images habve been normalized using: image = image - mean / std
        # to plot images we have to undo the normalization 
        if images_are_normalized:
            images = images * 0.5 + 0.5
        
        for img_indx, curr_image in enumerate(images):
        
            if len(samples[labels[img_indx]]) < num_samples:
                samples[labels[img_indx]].append(curr_image.numpy())

    # convert samples to numpy array
    return np.array(samples)


def plot_samples(full_dataloaders, classes_str, images_are_normalized: bool, filaname: str = 'samples', path_to_save: str = 'exploring_data', phase: str = 'train'):
    samples = get_samples(full_dataloaders[phase], 10, images_are_normalized)

    fig, axs = plt.subplots(10, 10, figsize=(10, 10), gridspec_kw=dict(hspace=0.0))

    for i in range(10):  
        for j in range(10):  
            image = np.transpose(samples[i][j], (1, 2, 0))  # Transpose the image to (32, 32, 3)
            axs[j, i].imshow(image)
            axs[j, i].axis('off')

        axs[0, i].title.set_text(classes_str[i])    

    plt.tight_layout()
    make_dir(f'{path_to_save}')
    plt.savefig(f'{path_to_save}/{filaname}.jpg')
    plt.clf()


# fdl, _, cstr, _ = get_cifar10_data(11)
# plot_samples(fdl, cstr, filaname='temp', images_are_normalized=False)


class SelectedClassesImageFolder(ImageFolder):
    def __init__(self, root, selected_classes, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.selected_classes = selected_classes
        self.class_to_idx = {class_name: i for i, class_name in enumerate(selected_classes)}
        self.idx_to_class = {i: class_name for i, class_name in enumerate(selected_classes)}
        self.temp_del = np.zeros(len(selected_classes))

        self.samples = self._filter_samples()

        
    def _filter_samples(self):
        filtered_samples = []
        for path, target in self.samples:
            class_name = self.classes[target]
            if class_name in self.selected_classes:
                filtered_samples.append((path, self.class_to_idx[class_name]))
                self.temp_del[self.class_to_idx[class_name]] += 1

        print(len(filtered_samples))
        print(self.temp_del)
        print(np.where(self.temp_del == 0.0)[0])
        print(self.temp_del.sum())
        

        return filtered_samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        return super().__getitem__(index)[0], target

    def __len__(self):
        return len(self.samples)
def get_domain_net(
                   group: str, 
                   normalize_data: bool,\
                   image_size: int, 
                   batch_size=128, 
                   randseed: int =11, pin_memory=False,
                   num_workers=1,
                   return_test_data: bool=True):
    
    train_domain = 'real'
    test_domain = 'painting'

    # assert domain == 'real' or domain == 'painting', 'only real and painting domain.'
    assert group == 'mammals', 'only mammals group of dataset'
    
    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(randseed)
        random.seed(randseed)

    g = torch.Generator()
    g.manual_seed(randseed)

    if not normalize_data:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    root = f'./DomainNet/{train_domain}/{group}/{train_domain}'
    
    # torchvision.datasets.ImageFolder
    dataset = torchvision.datasets.ImageFolder(
        root = root,
        transform = transform,
    )
    print(len(dataset))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            worker_init_fn=seed_worker,
                            generator=g,
                            shuffle=True,
                            pin_memory=pin_memory,
                            num_workers=num_workers)

    if return_test_data: 
        #### painting as test data
        root_test = f'./DomainNet/{test_domain}/{group}/{test_domain}'

        test_dataset = torchvision.datasets.ImageFolder(
            root = root_test,
            transform = transform,
        )
        print(len(test_dataset))
        
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                worker_init_fn=seed_worker,
                                generator=g,
                                shuffle=False,
                                pin_memory=pin_memory,
                                num_workers=num_workers)    

        return {'train': dataloader, 'test': test_dataloader}, list(dataset.class_to_idx.keys())
    
    else:
        return {'train': dataloader, 'test': None}, list(dataset.class_to_idx.keys())
    