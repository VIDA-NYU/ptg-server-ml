# PTG Server-side Machine Learning

## Install

```bash
# git clone (includes the model repos too)
git clone --recursive git@github.com:VIDA-NYU/ptg-server-ml.git
# install basic (no model dependencies)
pip install -e .

export MODEL_DIR=models  # where do you want to store the weights?
```

## EgoVLP

Zero-shot action recognition

[Repo](https://github.com/showlab/EgoVLP) | 
[Checkpoint (2GB)](https://drive.google.com/file/d/1-SOQeXc-xSn544sJzgFLhC95hkQsm0BR)

```bash
pip install -e '.[egovlp]'

# download the weights
pip install gdown
gdown https://drive.google.com/uc?id=1-SOQeXc-xSn544sJzgFLhC95hkQsm0BR
mv epic_mir_plus.pth $MODEL_DIR/

# to test with a mp4 file
python -m ptgprocess.egovlp ./video.mp4 recipe:coffee:steps_simple
```


## Detic

Zero-shot object detection

[Repo](https://github.com/facebookresearch/Detic) | 
[Checkpoint](https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth)

```bash
pip install -e '.[detic]'
# model downloads its own weights

# to test with a mp4 file
python -m ptgprocess.detic ./video.mp4 recipe:pinwheels:steps_simple
```

## EgoHOS

Hand-object interactions

[Repo](https://github.com/owenzlz/EgoHOS) | 
[Checkpoint (5GB)](https://drive.google.com/uc?id=1LNMQ6TGf1QaCjMgTExPzl7lFFs-yZyqX)

```bash
pip install -e '.[egohos]'

# download the weights
pip install gdown
gdown https://drive.google.com/uc?id=1LNMQ6TGf1QaCjMgTExPzl7lFFs-yZyqX
unzip work_dirs.zip && rm work_dirs.zip
mv work_dirs $MODEL_DIR/egohos

# to test with a mp4 file
python -m ptgprocess.egohos ./video.mp4
python -m ptgprocess.egohos ./video.mp4 --out-file # save to file
```
