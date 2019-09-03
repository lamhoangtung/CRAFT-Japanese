"""
Config containing all hardcoded parameters for training strong supervised model on synth-text
"""
from config import *

DataLoader_JPN_SYNTH_dataset_path = '/mnt/data/linus/SynthText/results/linus_JPN_2.h5'

num_cuda = "4"
save_path = './model'
use_cuda = True

batch_size = {
	'train': 4*len(num_cuda.split(',')),
	'test': 8*len(num_cuda.split(',')),
}

num_workers = {
	'train': 16,
	'test': 16
}

pretrained = False
pretrained_path = './model/63000_model.pkl'
pretrained_loss_plot_training = './logs/loss_plot_training.npy'

lr = {
	1: 1e-4,
	10000: 5e-5,
	20000: 2e-5,
	40000: 1e-5,
	60000: 1e-6,
}

num_epochs_strong_supervision = 2

periodic_fscore = 300
periodic_output = 3000
periodic_save = 3000
optimizer_iteration = 1

visualize_generated = False

weight_threshold = 0.5

model_architecture = 'craft'

image_size = [1024, 1024]
