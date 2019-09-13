import cv2
import os
import json
import glob
from tqdm import tqdm
from shutil import copyfile


def dump_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as label_file:
        json.dump(data, label_file, ensure_ascii=False, indent=4)
        label_file.write("\n")
    print('Saved', output_path)


def main(datapile_path, output_path='./input/datapile/'):
    dataset_name = [each for each in datapile_path.split('/') if each != ''][-1]
    print('Processing dataset {}'.format(dataset_name))
    images_path = os.path.join(datapile_path, 'images')
    labels_path = os.path.join(datapile_path, 'labels')

    samples_path = glob.glob(os.path.join(labels_path, '*.json'))
    if not samples_path:
        raise FileNotFoundError(
            "Can't found any .json label file")

    os.makedirs(output_path, exist_ok=True)
    images_output_path = os.path.join(output_path, 'images')
    os.makedirs(images_output_path, exist_ok=True)

    dataset = []
    tbar = tqdm(samples_path)
    for sample in tbar:
        file_name = sample.split('/')[-1].replace('.json', '')
        tbar.set_description(file_name)
        current_json = json.loads(open(sample, 'r', encoding='utf-8').read())
        current_json = current_json["attributes"]["_via_img_metadata"]
        if not current_json:
            print(sample, 'is empty. Skiped')
            continue
        current_image_path = os.path.join(
            images_path, sample.split('/')[-1].replace('.json', '.png'))
        if not os.path.isfile(current_image_path):
            print(current_image_path, 'not found. Skiped')
            continue
    
        current_image = cv2.imread(current_image_path)
        for index, each_region in enumerate(current_json['regions']):
            if each_region['shape_attributes']['name'] == 'rect':
                # Crop image
                x = each_region['shape_attributes']['x']
                y = each_region['shape_attributes']['y']
                width = each_region['shape_attributes']['width']
                height = each_region['shape_attributes']['height']
                line_img = current_image[y:y+height, x:x+width]
                output_name = '{}_{}_{}.jpg'.format(dataset_name, file_name, index)
                if 'label' not in each_region['region_attributes']:
                    # print('Not found label, skipped')
                    continue
                if each_region['region_attributes']['label'] == '':
                    # print('Empty OCR. Skipped')
                    continue

                index += 1
                cv2.imwrite(os.path.join(images_output_path, output_name), line_img)

                # print('Saved', output_name)
                # Sometimes the dict key might be `truelabel` or `true label`
                dataset.append(
                    {'image_path': output_name, 'label': each_region['region_attributes']['label']})

    print('Total cropped {} line for OCR.'.format(len(dataset)))
    output_label_path = os.path.join(output_path, '{}.json'.format(dataset_name))
    dump_to_json(data=dataset, output_path=output_label_path)


if __name__ == "__main__":
    datapile_path = '/Users/linus/Downloads/1567682282_Invoice_Toyota4_Training_20190821_2'
    main(datapile_path)
