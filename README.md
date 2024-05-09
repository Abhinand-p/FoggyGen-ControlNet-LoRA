# Foggy Image Generation with Stable Diffusion


## Description
This repository is part of the final project for the course `Computer Vision: 3D Reconstruction`.

## Installation

### Requirements

Tested with CUDA 12.2 on Ubuntu 22.04

a. Create a conda environment and activate it.

```shell
conda create -n finalproject3dcv python
conda activate finalproject3dcv

```

b. Follow the [official instructions](https://pytorch.org/) to install Pytorch.

c. Clone the diffusers repo and install the library (the pip package is not sufficient).

```shell
git clone https://github.com/huggingface/diffusers/
cd diffusers
pip install .
```

d. Clone this repo and install all other requirements.

```shell
cd ..
git clone https://github.com/llswrtn/FinalProject3DCV
cd FinalProject3DCV

pip install -r requirements.txt
```


c. To run the `demo.ipynb` notebook, create a directory `weights`, download a [trained LoRA model](https://drive.google.com/drive/folders/1rTomopvoKfo1jFGK21pdlEa8uNZjf_Tx?usp=sharing) and place it into the newly created `weights` directory. When using a different LoRA model or depth map, adapt the paths in the second cell of the notebook accordingly.

For evaluation, please use the `eval/fid.py` script. This was not included into the demo notebook, since the computation of the FID score requires a large number of generated images and considerably more computational resources. 