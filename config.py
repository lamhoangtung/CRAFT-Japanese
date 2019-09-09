import os

seed = 0

dataset_name = 'datapile'
test_dataset_name = 'ICDAR2013'

DataLoader_JPN_SYNTH_dataset_path = './linus_JPN_2.h5'
DataLoader_JPN_SYNTH_Train_Synthesis = '/mnt/data/linus/CRAFT-Remade/logs/train_synthesis/'

# DataLoaderSYNTH_base_path = '/home/SharedData/Mayank/SynthText/Images'
# DataLoaderSYNTH_mat = '/home/SharedData/Mayank/SynthText/gt.mat'
# DataLoaderSYNTH_Train_Synthesis = '/home/SharedData/Mayank/Models/SYNTH/train_synthesis/'

DataLoader_Other_Synthesis = '/home/SharedData/Mayank/'+dataset_name+'/Save/'
Other_Dataset_Path = os.path.join('./input/', dataset_name)
save_path = '/home/SharedData/Mayank/Models/WeakSupervision/'+dataset_name
images_path = '/home/SharedData/Mayank/'+dataset_name+'/Images'
target_path = os.path.join('./input', dataset_name, 'generated')

# Test_Dataset_Path = '/home/SharedData/Mayank/'+test_dataset_name

threshold_character = 0.5
threshold_affinity = 0.5
threshold_word = 0.7
threshold_fscore = 0.5

dataset_pre_process = {
	'ic13': {
		'train': {
			'target_json_path': None,
			'target_folder_path': None,
		},
		'test': {
			'target_json_path': None,
			'target_folder_path': None,
		}
	},
	'ic15': {
		'train': {
			'target_json_path': None,
			'target_folder_path': None,
		},
		'test': {
			'target_json_path': None,
			'target_folder_path': None,
		}
	},
	'datapile': {
		'train': {
			'target_json_name': 'train_gt.json',
			'base_path': './input/datapile/',
		},
		'test': {
			'target_json_path': None,
			'target_folder_path': None,
		}
	}
}
