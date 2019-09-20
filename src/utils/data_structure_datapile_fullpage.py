import json
import os
import glob
import cv2
from tqdm import tqdm


def preprocess_datapile_fullpage(base_path, output_name):
    craft_anno = {}
    craft_anno["unknown"] = "###"
    craft_anno["annots"] = {}
    samples_path = glob.glob(os.path.join(base_path, 'labels', '*.json'))
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
            base_path, 'images', sample.split('/')[-1].replace('.json', '.png'))
        image_file_name = current_image_path.split('/')[-1]
        if not os.path.isfile(current_image_path):
            print(current_image_path, 'not found. Skiped')
            continue
        bbox = []
        text = []
        for index, each_region in enumerate(current_json['regions']):
            if each_region['shape_attributes']['name'] == 'rect':
                x = each_region['shape_attributes']['x']
                y = each_region['shape_attributes']['y']
                width = each_region['shape_attributes']['width']
                height = each_region['shape_attributes']['height']
                line_bb = [[x, y], [width, y], [width, height], [x, height]]
                bbox.append(line_bb)
                text.append(each_region['region_attributes']['label'])
        if image_file_name not in craft_anno["annots"]:
            craft_anno["annots"][image_file_name] = {}
        craft_anno["annots"][image_file_name]['bbox'] = bbox
        craft_anno['annots'][image_file_name]['text'] = text

    with open(os.path.join(base_path, output_name), 'w', encoding='utf-8') as f:
        json.dump(craft_anno, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    preprocess_datapile_fullpage('./input/toyota/', 'train_gt.json')
