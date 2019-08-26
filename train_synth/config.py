"""
Config containing all hardcoded parameters for training strong supervised model on synth-text
"""

num_cuda = "0"
seed = 0
save_path = './model'
use_cuda = False

batch_size = {
	'train': 4,
	'test': 3,
}

pretrained = False
pretrained_path = ''
pretrained_loss_plot_training = 'model/loss_plot_training.npy'

lr = {
	1: 5e-5,
	30000: 2.5e-5,
	60000: 1e-5,
	120000: 5e-6,
	180000: 1e-6,
}

periodic_fscore = 300
periodic_output = 3000
periodic_save = 30000

threshold_character = 0.4
threshold_affinity = 0.4
threshold_word = 0.7
threshold_fscore = 0.5

DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

ICDAR2013_path = '/home/SharedData/Mayank/ICDAR2015'

visualize_generated = False


def get_weight_threshold(min_, max_, iteration):
	import numpy as np
	weight_threshold_ = []
	for i in range(iteration-1):
		weight_threshold_.append(min_ + i*(max_ - min_)/(iteration-1))

	weight_threshold_.append(max_)
	return np.flip(np.array(weight_threshold_))


weight_threshold = get_weight_threshold(0.5, 0.5, 20)

model_architecture = 'craft'
