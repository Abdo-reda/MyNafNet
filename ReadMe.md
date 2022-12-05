# Deep Learning Project - Image Denosing - NafNet Model


## Inference
For Inference, please use the following [Notebook](https://colab.research.google.com/drive/1BjQfvzMe9aRvhlItpOCdoCet9XNBJvsE?usp=sharing)


## Training

### Download Dataset

##### Download the train set and place it in ```./datasets/SIDD/Data```:

* [google drive](https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing)
* ```python scripts/data_preparation/sidd.py``` to crop the train image pairs to 512x512 patches and make the data into lmdb format.

##### Download the evaluation data (in lmdb format) and place it in ```./datasets/SIDD/val/```:

  * [google drive](https://drive.google.com/file/d/1gZx_K2vmiHalRNOb1aj93KuUQ2guOlLp/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1I9N5fDa4SNP0nuHEy6k-rw?pwd=59d7), 
  * it should be like ```./datasets/SIDD/val/input_crops.lmdb``` and ```./datasets/SIDD/val/gt_crops.lmdb``

### Setup Enviornment

    python3 venv env
    source env/bin/activate
    pip3 install -r requirements_train.txt


#### List of Dependencies:
* addict
* future
* lmdb
* numpy
* opencv-python
* Pillow
* pyyaml
* requests
* scikit-image
* scipy
* tb-nightly
* tqdm
* yapf
* torch
* torchvision

### Train the Model

    python -m torch.distributed.launch NAFNet/basicsr/train.py -opt NAFNet/options/train/SIDD/NAFNet-width32.yml --launcher pytorch


### Notes
- Make sure to download the correct version of torch for your system/gpu.
- Make sure to change the files that point to the dataset directory, (Options file.)