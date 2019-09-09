import json
import os

import cv2


def preprocess_datapile(
        base_path='./input/datapile/',
        output_path='train_gt.json'):
    """
    This function converts the Datapile images to the format which we need
    to train our weak supervision model.
    More data set conversion functions would be written here
    :param base_path: Put your path to the datapile data set
    :param output_path: Will convert the ground truth to a json format at the location output_path
    :return: None
    """

    raw_dataset = json.loads(
        open(os.path.join(base_path, 'label.json'), 'r').read())
    all_annots = {'unknown': '###', 'annots': {}}

    for sample in raw_dataset:
        image_name = sample['image_path'].split('/')[-1].replace('.jpg', '')
        all_annots['annots'][image_name] = {}
        # print(os.path.join(base_path, 'images', sample['image_path']))
        height, width, _ = cv2.imread(os.path.join(
            base_path, 'images', sample['image_path'])).shape
        line_bb = [[0, 0], [0, width], [height, width], [height, 0]]
        all_annots['annots'][image_name]['bbox'] = line_bb
        all_annots['annots'][image_name]['text'] = sample['label']

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_annots, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    preprocess_datapile()
