# PTG Server-side Machine Learning

## Install

```bash
# install basic (no model dependencies)
pip install -e .
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
unzip work_dirs.zip && rm work_dirs.zip
mv work_dirs $MODEL_DIR/egohos
```


## Detic

Zero-shot object detection

[Repo](https://github.com/facebookresearch/Detic) | 
[Checkpoint](https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth)

```bash
pip install -e '.[detic]'
# model downloads its own weights
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
```