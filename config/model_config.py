############### configuration file ###############
import numpy as np

import torchvision.transforms as transforms
import utils.mytransforms as mytransforms

#- Augmentation -#
train_transform = {
            'imagenet': 
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                mytransforms.Lighting(0.1, mytransforms.IMAGENET_PCA['eigval'], mytransforms.IMAGENET_PCA['eigvec']),
                transforms.Normalize(mytransforms.IMAGENET_STATS['mean'], mytransforms.IMAGENET_STATS['std'])
                ]),
            'inat': 
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.INAT_STATS['mean'], mytransforms.INAT_STATS['std'])
            ]),
}
test_transform = {
            'imagenet':
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.IMAGENET_STATS['mean'], mytransforms.IMAGENET_STATS['std'])
                ]),
            'inat': 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mytransforms.INAT_STATS['mean'], mytransforms.INAT_STATS['std'])
            ]),
}

#----------Learning Rate-------------#
cos_lr = lambda lr, T: ( 0.5*lr*np.cos(np.pi*np.arange(T)/T) + 0.5*lr ).tolist()

ResNet50Feature = {
    'arch' : 'ResNet50Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.2, 90),
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet50Feature_finetune = {
    'arch' : 'ResNet50Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.2, 30),
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet10Feature = {
    'arch' : 'ResNet10Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.2, 90),
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}

ResNet10Feature_finetune = {
    'arch' : 'ResNet10Feature',
    'batch_size' : 256,
    'lrs' : cos_lr(0.2, 30),
    'opt_params': {'weight_decay' : 0.0001, 'momentum': 0.9},
}
