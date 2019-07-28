num_cuda = "0,1,2,3,4,5"
seed = 0
save_path = 'model'
use_cuda = True

batchsize = {
	'train': 18,
	'test': 1
}

DEBUG = False
if DEBUG:
	batchsize['train'] = 1
	num_cuda = '0'

pretrained = True
pretrained_path = '/home/Krishna.Wadhwani/Dataset/Programs/CRAFT-Remade/Stage-1/model/41958_model.pkl'
pretrained_path_test = '/home/Krishna.Wadhwani/Dataset/Programs/CRAFT-Remade/Stage-1/model/final_model.pkl'

lr = {
	1: 5e-5,
	10000: 9e-4,
	20000: 5e-4,
	30000: 1e-4,
}

periodic_fscore = 100
periodic_output = 1000
periodic_save = 999

threshold_character = 0.5
threshold_affinity = 0.5
threshold_fscore = 0.5
threshold_first_character = 0.5
threshold_boundary = 0.5
threshold_pos = 0.1