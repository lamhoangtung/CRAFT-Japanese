import json
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

import train_weak_supervision.config as config
from src.utils.data_manipulation import (generate_affinity, generate_target,
                                         generate_target_others,
                                         normalize_mean_variance, resize,
                                         resize_generated)


class DataLoaderMIX_JPN(data.Dataset):

    """
            Dataloader to train weak-supervision providing a mix of SynthText JPN and the Datapile's dataset
    """

    def __init__(self, type_, iteration):
        self.DEBUG = True
        self.type_ = type_
        self.dataset_path = config.DataLoader_JPN_SYNTH_dataset_path
        self.raw_dataset = h5py.File(self.dataset_path, 'r')
        self.ids = self.__get_list_id__()
        self.base_path_other_images = os.path.join(
            config.Other_Dataset_Path, type_, 'images')
        self.base_path_other_gt = os.path.join(
            config.Other_Dataset_Path, 'generated', str(iteration))

        if config.prob_synth != 0:
            if self.DEBUG:
                self.ids = self.ids[:5]

            total_number = len(self.ids)
            train_images = int(total_number * 0.9)
            print('Training with', train_images, 'images')

            if self.type_ == 'train':
                self.imnames = self.ids[:train_images]
            else:
                self.imnames = self.ids[train_images:]

        self.gt = []

        for i in sorted(os.listdir(self.base_path_other_gt)):
            with open(self.base_path_other_gt+'/'+i, 'r') as f:
                self.gt.append([i[:-5], json.load(f)])
        if self.DEBUG:
            self.gt = self.gt[:5]

    def __get_list_id__(self):
        return [file_id for file_id in self.raw_dataset['data']]

    def __getitem__(self, item_i):
        height, width, channel = 0, 0, 0
        # noinspection PyArgumentList
        np.random.seed()
        check = np.random.uniform()

        if check < config.prob_synth and self.type_ == 'train':
            # probability of picking a Synth-Text image vs Image from dataset

            random_item = np.random.choice(self.imnames)
            sample = self.raw_dataset['data'][random_item]
            image = sample[()]
            charBB = sample.attrs['charBB']
            txt = [each.decode('utf-8') for each in sample.attrs['txt']]
            # print(txt)
            # Handle line-break
            all_words = []
            for line in txt:
                if '\n' in line:
                    all_words.extend(line.split('\n'))
                else:
                    all_words.append(line)
            # Remove blank word
            for index, line in enumerate(all_words):
                all_words[index] = [word for word in line.strip().split(' ')
                                    if word not in ['', ' ']]
            # Split word to char
            for index, line in enumerate(all_words):
                new_line = []
                for word in line:
                    if len(word) >= 2:
                        new_line.extend([char for char in word])
                    else:
                        new_line.append(word)
                all_words[index] = new_line
            # print('--------')
            # print(all_words)
            # print('--------')

            # Resize the image to (768, 768)
            image, character = resize(image, charBB.copy())
            image = normalize_mean_variance(image).transpose(2, 0, 1)

            # Generate character heatmap with weights
            weight_character, weak_supervision_char = generate_target(
                image.shape, character.copy(), weight=1)

            # Generate affinity heatmap with weights
            weight_affinity, weak_supervision_affinity = generate_affinity(
                image.shape, character.copy(),
                all_words.copy(),
                weight=1)

            dataset_name = 'SYNTH'
            text_target = ''

        else:

            random_item = np.random.randint(len(self.gt))
            image = plt.imread(os.path.join(
                self.base_path_other_images, self.gt[random_item][0]))  # Read the image

            if len(image.shape) == 2:
                image = np.repeat(image[:, :, None], repeats=3, axis=2)
            elif image.shape[2] == 1:
                image = np.repeat(image, repeats=3, axis=2)
            else:
                image = image[:, :, 0: 3]

            height, width, channel = image.shape
            character = [
                np.array(word_i).reshape([len(word_i), 4, 1, 2]) for word_i in self.gt[random_item][1]['characters'].copy()]
            affinity = [
                np.array(word_i).reshape([len(word_i), 4, 1, 2]) for word_i in self.gt[random_item][1]['affinity'].copy()]

            assert len(character) == len(
                affinity), 'word length different in character and affinity'

            # Resize the image to (768, 768)
            image, character, affinity = resize_generated(
                image, character.copy(), affinity.copy())
            image = normalize_mean_variance(image).transpose(2, 0, 1)
            weights = [i for i in self.gt[random_item][1]['weights'].copy()]
            text_target = '#@#@#@'.join(self.gt[random_item][1]['text'])

            assert len(self.gt[random_item][1]['text']) == len(self.gt[random_item][1]['word_bbox']), \
                'Length of word_bbox != Length of text'

            # assert len(text_target.split('#@#@#@')) == len(self.gt[random_item][1]['word_bbox']), \
            # 	'Some error in splitting'

            # Generate character heatmap with weights
            weight_character, weak_supervision_char = generate_target_others(
                image.shape, character.copy(), weights.copy())

            # Generate affinity heatmap with weights
            weight_affinity, weak_supervision_affinity = generate_target_others(
                image.shape, affinity.copy(), weights.copy())

            # Get original word_bbox annotations
            dataset_name = 'datapile'


        return \
            image.astype(np.float32), \
            weight_character.astype(np.float32), \
            weight_affinity.astype(np.float32), \
            weak_supervision_char.astype(np.float32), \
            weak_supervision_affinity.astype(np.float32), \
            dataset_name, \
            text_target, \
            str(random_item), \
            np.array([height, width])

    def __len__(self):

        if self.type_ == 'train':

            return config.iterations
        else:

            return len(self.gt)


class DataLoaderMIX(data.Dataset):

    """
            Dataloader to train weak-supervision providing a mix of SynthText and the dataset
    """

    def __init__(self, type_, iteration):

        self.type_ = type_
        self.base_path_synth = config.DataLoaderSYNTH_base_path
        self.base_path_other_images = config.Other_Dataset_Path + '/Images/' + type_
        self.base_path_other_gt = config.Other_Dataset_Path + \
            '/Generated/' + str(iteration)

        if config.prob_synth != 0:

            print('Loading Synthetic dataset')

            if DEBUG:  # Make this True if you want to do a run on small set of Synth-Text
                if not os.path.exists('cache.pkl'):

                    # Create cache of 1000 samples if it does not exist

                    with open('cache.pkl', 'wb') as f:
                        import pickle
                        from scipy.io import loadmat
                        mat = loadmat(config.DataLoaderSYNTH_mat)
                        pickle.dump(
                            [mat['imnames'][0][0:1000], mat['charBB'][0][0:1000], mat['txt'][0][0:1000]], f)
                        print('Created the pickle file, rerun the program')
                        exit(0)
                else:

                    # Read the Cache

                    with open('cache.pkl', 'rb') as f:
                        import pickle
                        self.imnames, self.charBB, self.txt = pickle.load(f)
                        print('Loaded DEBUG')

            else:

                from scipy.io import loadmat
                # Loads MATLAB .mat extension as a dictionary of numpy arrays
                mat = loadmat(config.DataLoaderSYNTH_mat)

                # Read documentation of how synth-text dataset is stored to understand the processing at
                # http://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt

                total_number = mat['imnames'][0].shape[0]
                train_images = int(total_number * 0.9)

                if self.type_ == 'train':

                    self.imnames = mat['imnames'][0][0:train_images]
                    # number of images, 2, 4, num_character
                    self.charBB = mat['charBB'][0][0:train_images]
                    self.txt = mat['txt'][0][0:train_images]

                else:

                    self.imnames = mat['imnames'][0][train_images:]
                    # number of images, 2, 4, num_character
                    self.charBB = mat['charBB'][0][train_images:]
                    self.txt = mat['txt'][0][train_images:]

            for no, i in enumerate(self.txt):
                all_words = []
                for j in i:
                    all_words += [k for k in ' '.join(
                        j.split('\n')).split() if k != '']
                    # Getting all words given paragraph like text in SynthText

                self.txt[no] = all_words

        self.gt = []

        for i in sorted(os.listdir(self.base_path_other_gt)):
            with open(self.base_path_other_gt+'/'+i, 'r') as f:
                self.gt.append([i[:-5], json.load(f)])

    def __getitem__(self, item_i):

        # noinspection PyArgumentList
        np.random.seed()
        check = np.random.uniform()

        if check < config.prob_synth and self.type_ == 'train':
            # probability of picking a Synth-Text image vs Image from dataset

            random_item = np.random.randint(len(self.imnames))

            character = self.charBB[random_item].copy()

            image = plt.imread(self.base_path_synth+'/' +
                               self.imnames[random_item][0])  # Read the image

            if len(image.shape) == 2:
                image = np.repeat(image[:, :, None], repeats=3, axis=2)
            elif image.shape[2] == 1:
                image = np.repeat(image, repeats=3, axis=2)
            else:
                image = image[:, :, 0: 3]

            height, width, channel = image.shape
            # Resize the image to (768, 768)
            image, character = resize(image, character)
            image = normalize_mean_variance(image).transpose(2, 0, 1)

            # Generate character heatmap with weights
            weight_character, weak_supervision_char = generate_target(
                image.shape, character.copy(), weight=1)

            # Generate affinity heatmap with weights
            weight_affinity, weak_supervision_affinity = generate_affinity(
                image.shape, character.copy(),
                self.txt[random_item].copy(),
                weight=1)

            dataset_name = 'SYNTH'
            text_target = ''

        else:

            random_item = np.random.randint(len(self.gt))
            image = plt.imread(self.base_path_other_images +
                               '/'+self.gt[random_item][0])  # Read the image

            if len(image.shape) == 2:
                image = np.repeat(image[:, :, None], repeats=3, axis=2)
            elif image.shape[2] == 1:
                image = np.repeat(image, repeats=3, axis=2)
            else:
                image = image[:, :, 0: 3]

            height, width, channel = image.shape
            character = [
                np.array(word_i).reshape([len(word_i), 4, 1, 2]) for word_i in self.gt[random_item][1]['characters'].copy()]
            affinity = [
                np.array(word_i).reshape([len(word_i), 4, 1, 2]) for word_i in self.gt[random_item][1]['affinity'].copy()]

            assert len(character) == len(
                affinity), 'word length different in character and affinity'

            # Resize the image to (768, 768)
            image, character, affinity = resize_generated(
                image, character.copy(), affinity.copy())
            image = normalize_mean_variance(image).transpose(2, 0, 1)
            weights = [i for i in self.gt[random_item][1]['weights'].copy()]
            text_target = '#@#@#@'.join(self.gt[random_item][1]['text'])

            assert len(self.gt[random_item][1]['text']) == len(self.gt[random_item][1]['word_bbox']), \
                'Length of word_bbox != Length of text'

            # assert len(text_target.split('#@#@#@')) == len(self.gt[random_item][1]['word_bbox']), \
            # 	'Some error in splitting'

            # Generate character heatmap with weights
            weight_character, weak_supervision_char = generate_target_others(
                image.shape, character.copy(), weights.copy())

            # Generate affinity heatmap with weights
            weight_affinity, weak_supervision_affinity = generate_target_others(
                image.shape, affinity.copy(), weights.copy())

            # Get original word_bbox annotations
            dataset_name = 'ICDAR'

        return \
            image.astype(np.float32), \
            weight_character.astype(np.float32), \
            weight_affinity.astype(np.float32), \
            weak_supervision_char.astype(np.float32), \
            weak_supervision_affinity.astype(np.float32), \
            dataset_name, \
            text_target, \
            random_item, \
            np.array([height, width])

    def __len__(self):

        if self.type_ == 'train':

            return config.iterations
        else:

            return len(self.gt)


class DataLoaderEvalOther_Datapile(data.Dataset):

    """
            Datapile dataloader
    """

    def __init__(self, type_):

        self.type_ = type_
        self.base_path = os.path.join(
            config.Other_Dataset_Path, type_, 'images')

        with open(os.path.join(config.Other_Dataset_Path, type_, '{}_gt.json'.format(self.type_)), 'r') as f:
            self.gt = json.load(f)

        self.imnames = sorted(self.gt['annots'].keys())
        self.unknown = self.gt['unknown']

        self.debug = True
        if self.debug:
            self.imnames = self.imnames[:5]

    def __getitem__(self, item):
        """
        Function to read, resize and pre-process the image from the datapile dataset
        :param item:
        :return:
        """
        image = plt.imread(os.path.join(self.base_path, self.imnames[item]))

        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], repeats=3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, repeats=3, axis=2)
        else:
            image = image[:, :, 0: 3]

        height, width, channel = image.shape
        max_side = max(height, width)
        new_resize = (int(width / max_side * 768),
                      int(height / max_side * 768))
        image = cv2.resize(image, new_resize)

        big_image = np.ones([768, 768, 3], dtype=np.float32) * np.mean(image)
        big_image[
            (768 - image.shape[0]) // 2: (768 - image.shape[0]) // 2 + image.shape[0],
            (768 - image.shape[1]) // 2: (768 - image.shape[1]) // 2 + image.shape[1]] = image
        big_image = normalize_mean_variance(big_image).transpose(2, 0, 1)

        return big_image.astype(np.float32), self.imnames[item], np.array([height, width]), item

    def __len__(self):

        return len(self.imnames)


class DataLoaderEvalOther(data.Dataset):

    """
            ICDAR 2013 dataloader
    """

    def __init__(self, type_):

        self.type_ = type_
        if self.type_ == 'train':
            self.base_path = config.Other_Dataset_Path + '/Images/'
        else:
            self.base_path = config.Test_Dataset_Path + '/Images/'

        with open(self.base_path + self.type_ + '_gt.json', 'r') as f:
            self.gt = json.load(f)

        self.imnames = sorted(self.gt['annots'].keys())
        self.unknown = self.gt['unknown']

    def __getitem__(self, item):
        """
        Function to read, resize and pre-process the image from the icdar 2013 dataset
        :param item:
        :return:
        """

        image = plt.imread(self.base_path+self.type_+'/'+self.imnames[item])

        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], repeats=3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, repeats=3, axis=2)
        else:
            image = image[:, :, 0: 3]

        height, width, channel = image.shape
        max_side = max(height, width)
        new_resize = (int(width / max_side * 768),
                      int(height / max_side * 768))
        image = cv2.resize(image, new_resize)

        big_image = np.ones([768, 768, 3], dtype=np.float32) * np.mean(image)
        big_image[
            (768 - image.shape[0]) // 2: (768 - image.shape[0]) // 2 + image.shape[0],
            (768 - image.shape[1]) // 2: (768 - image.shape[1]) // 2 + image.shape[1]] = image
        big_image = normalize_mean_variance(big_image).transpose(2, 0, 1)

        return big_image.astype(np.float32), self.imnames[item], np.array([height, width]), item

    def __len__(self):

        return len(self.imnames)
