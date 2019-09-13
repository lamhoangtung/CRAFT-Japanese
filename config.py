import os
import math

seed = 0

dataset_name = 'datapile'
test_dataset_name = 'datapile'

DataLoader_JPN_SYNTH_dataset_path = './input/synth_text/linus_JPN_2.h5'
DataLoader_JPN_SYNTH_Train_Synthesis = './logs/train_synthesis/'


THRESHOLD_POSITIVE = 0.1
THRESHOLD_NEGATIVE = 0

threshold_point = 25
window = 120

sigma = 18.5
sigma_aff = 20

boundary_character = math.exp(-1/2*(threshold_point**2)/(sigma**2))
boundary_affinity = math.exp(-1/2*(threshold_point**2)/(sigma_aff**2))

threshold_character = boundary_character + 0.03
threshold_affinity = boundary_affinity + 0.03

threshold_character_upper = boundary_character + 0.2
threshold_affinity_upper = boundary_affinity + 0.2

scale_character = math.sqrt(math.log(boundary_character)/math.log(threshold_character_upper))
scale_affinity = math.sqrt(math.log(boundary_affinity)/math.log(threshold_affinity_upper))


print(
	'Boundary character value = ', boundary_character,
	'| Threshold character value = ', threshold_character,
	'| Threshold character upper value = ', threshold_character_upper
)
print(
	'Boundary affinity value = ', boundary_affinity,
	'| Threshold affinity value = ', threshold_affinity,
	'| Threshold affinity upper value = ', threshold_affinity_upper
)
print('Scale character value = ', scale_character, '| Scale affinity value = ', scale_affinity)
print('Training Dataset = ', dataset_name, '| Testing Dataset = ', test_dataset_name)

Other_Dataset_Path = os.path.join('./input/', dataset_name)
save_path = os.path.join('./model/weak_supervision/', dataset_name)
images_path = '/home/SharedData/Mayank/'+dataset_name+'/Images'
target_path = os.path.join('./input', dataset_name, 'generated')

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
			'base_path': './input/datapile/train/',
		},
		'test': {
			'target_json_name': 'test_gt.json',
			'base_path': './input/datapile/test/',
		}
	}
}

start_iteration = 0
skip_iterations = []