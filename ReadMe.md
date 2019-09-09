# Re-Implementing CRAFT-Character Region Awareness for Text Detection
Focused on Japanese Text

## Objective

- [X] Reproduce weak-supervision training as mentioned in the paper https://arxiv.org/pdf/1904.01941.pdf
- [ ] Generate character bbox on all Datapile's data sets.


## Clone the repository

    git clone https://github.com/autonise/CRAFT-Remade.git
    cd CRAFT-Remade

### Option 1: Conda Environment Installation
    conda env create -f environment.yml
    conda activate craft

### Option 2: Pip Installation
    pip3 install -r requirements.txt

## Running on custom images

- Put the images inside a folder.
- Get a pre-trained model from the pre-trained model list (Currently only strong supervision using SYNTH-Text available)
- Run the command
```bash
python3 main.py synthesize --model=./model/final_model.pkl --folder=./input
```
## Pre-trained models

### Strong Supervision

- SynthText(CRAFT Model) - [download here](https://drive.google.com/file/d/1be4MtJMEcaolM-s4EMsCRUmJFg2pR2OI/view?usp=sharing)
- SynthText(ResNet-UNet Model) - comming
- Original Model by authors - [download here](https://drive.google.com/open?id=1ZQE0tK9498RhLcXwYRgod4upmrYWdgl9)

### Weak Supervision

- [ ] Datapile - In Progress

## How to train the model from scratch

### Strong Supervision on Synthetic dataset

- Download the pre-trained model on Synthetic dataset at [here](https://drive.google.com/file/d/1be4MtJMEcaolM-s4EMsCRUmJFg2pR2OI/view?usp=sharing)
- Otherwise if you want to train from scratch
- Download my generated Japanese SynthText dataset at [here](https://drive.google.com/drive/folders/10WySKiBnO8WFU53Pt2QBQgY_GCBSZKWs?usp=sharing)
- Run the command
```bash
python3 main.py train_synth
```
- To test your model on SynthText, Run the command
```bash
python3 main.py test_synth --model /path/to/model
```
### Weak Supervision

#### First Pre-Process your dataset

- The assumed structure of the dataset is
```
.
├── generated (This folder will contain the weak-supervision intermediate targets)
├── train
│   ├── img_1.jpg
│   ├── img_2.jpg
│   ├── img_3.jpg
│   ├── img_4.jpg
│   └── img_5.jpg
│   └── ...
│   └── train_gt.json (This can be generated using the pre_process function described below)
├── test
│   ├── img_1.jpg
│   ├── img_2.jpg
│   ├── img_3.jpg
│   ├── img_4.jpg
│   └── img_5.jpg
│   └── ...
│   └── test_gt.json (This can be generated using the pre_process function described below)
```
- First convert datapile dataset to OCR only format using [datapile_to_onmt.py](src/utils/datapile_to_onmt.py) script
- To generate the json files for Datapile
```
In config.py change the corresponding values

'datapile': {
    'train': {
        'target_json_name': 'train_gt.json',
        'base_path': './input/datapile/train/',
    },
    'test': {
        'target_json_name': 'test_gt.json',
        'base_path': './input/datapile/test/',
    }
```
- Run the command:
```bash
python3 main.py pre_process --dataset datapile
```

#### Second Train your model based on weak-supervision
- Run the command

```bash
python3 main.py weak_supervision --model /path/to/strong/supervision/model --iterations <num_of_iterations(20)>
```

- This will train the weak supervision model for the number of iterations you specified
