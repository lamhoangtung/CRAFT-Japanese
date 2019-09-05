import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_synth.config as config
from src.utils.data_manipulation import denormalize_mean_variance
from train_synth import config as config
from train_synth.dataloader import DataLoader_JPN_SYNTH

train_dataloader = DataLoader_JPN_SYNTH('train')
train_dataloader = DataLoader(
    train_dataloader, batch_size=1,
    shuffle=False, num_workers=1)


def save(data, target, target_affinity, no):
    """
    Saving the synthesised outputs in between the training
    :param data: image as tensor
    :param target: character heatmap target as tensor
    :param target_affinity: affinity heatmap target as tensor
    :param no: current iteration number
    :return: None
    """

    data = data.data.cpu().numpy()
    target = target.data.cpu().numpy()
    target_affinity = target_affinity.data.cpu().numpy()

    base = './debug/'+str(no)+'/'

    os.makedirs(base, exist_ok=True)

    for i in range(1):

        os.makedirs(base+str(i), exist_ok=True)

        plt.imsave(base+str(i) + '/image.png',
                   denormalize_mean_variance(data[i].transpose(1, 2, 0)))
        plt.imsave(base+str(i) + '/target_characters.png',
                   target[i, :, :], cmap='gray')
        plt.imsave(base+str(i) + '/target_affinity.png',
                   target_affinity[i, :, :], cmap='gray')


iterator = tqdm(train_dataloader)
for no, (image, weight, weight_affinity) in enumerate(iterator):
    save(image, weight, weight_affinity, no)
