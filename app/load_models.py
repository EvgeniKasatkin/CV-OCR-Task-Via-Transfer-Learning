# -*- coding: utf-8 -*-
from IPython.core.interactiveshell import InteractiveShell
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
from PIL import Image
from torchsummary import summary
from timeit import default_timer as timer

InteractiveShell.ast_node_interactivity = 'all'

    
    
__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']



class VGG(nn.Module):
    
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    
    
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
    
    
def _vgg(arch, cfg, batch_norm, pretrained, progress = True, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = model.load_state_dict(torch.load('/terminal_recognition_last/NETWORKS/vgg16-397923af.pth'))
        #model.load_state_dict(state_dict)
    return model
    
    
    
def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
    
    
    
def load_checkpoint_assortment(path):

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Incorrect file name"
    checkpoint = torch.load('/recognition/NETWORKS/vgg16-multiclass_1_2.h5', map_location=torch.device('cpu'))

    if model_name == 'vgg16':
        model_assortment = vgg16(pretrained=True)

        for param in model_assortment.parameters():
            param.requires_grad = False
        model_assortment.classifier = checkpoint['classifier']

    model_assortment.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model_assortment.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_assortment.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    model_assortment.class_to_idx = checkpoint['class_to_idx']
    model_assortment.idx_to_class = checkpoint['idx_to_class']
    model_assortment.epochs = checkpoint['epochs']
    optimizer_assortment = checkpoint['optimizer']
    optimizer_assortment.load_state_dict(checkpoint['optimizer_state_dict'])
    return model_assortment, optimizer_assortment
    
    
    
def load_checkpoint_promo(path):

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Incorrect file name"

    checkpoint = torch.load('/recognition/NETWORKS/vgg16-promo_1002.h5', map_location=torch.device('cpu'))

    if model_name == 'vgg16':
        model_promo = vgg16(pretrained=True)
        for param in model_promo.parameters():
            param.requires_grad = False
        model_promo.classifier = checkpoint['classifier']

    model_promo.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model_promo.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_promo.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    #if multi_gpu:
    #    model = nn.DataParallel(model)
    #if train_on_gpu:
    #    model = model.to('cuda')

    model_promo.class_to_idx = checkpoint['class_to_idx']
    model_promo.idx_to_class = checkpoint['idx_to_class']
    model_promo.epochs = checkpoint['epochs']

    optimizer_promo = checkpoint['optimizer']
    optimizer_promo.load_state_dict(checkpoint['optimizer_state_dict'])

    return model_promo, optimizer_promo
    
    
    
    
def load_checkpoint_exist(path):

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Incorrect file name"
    checkpoint = torch.load('/recognition/NETWORKS/vgg16-exists.h5', map_location=torch.device('cpu'))

    if model_name == 'vgg16':
        model_exist = vgg16(pretrained=True)
        for param in model_exist.parameters():
            param.requires_grad = False
        model_exist.classifier = checkpoint['classifier']

    model_exist.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model_exist.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model_exist.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    model_exist.class_to_idx = checkpoint['class_to_idx']
    model_exist.idx_to_class = checkpoint['idx_to_class']
    model_exist.epochs = checkpoint['epochs']

    optimizer_exist = checkpoint['optimizer']
    optimizer_exist.load_state_dict(checkpoint['optimizer_state_dict'])

    return model_exist, optimizer_exist
    
    
    checkpoint_path = 'vgg16-multiclass_1_2.h5'
    model_assortment, optimizer_assortment = load_checkpoint_assortment(path=checkpoint_path)
    checkpoint_path = 'vgg16-promo_1002.h5'
    model_promo, optimizer_promo = load_checkpoint_promo(path=checkpoint_path)
    checkpoint_path = 'vgg16-exists.h5'
    model_exist, optimizer_exist = load_checkpoint_exist(path=checkpoint_path)
    return model_assortment, optimizer_assortment, model_promo, optimizer_promo, model_exist, optimizer_exist






