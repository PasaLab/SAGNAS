#!/usr/bin/env bash

conda create --name <virtualenv>
source activate base
conda activate <virtualenv>
conda install -y pytorch=1.13.0 torchvision cudatoolkit=11.0 python=3.7.16 Pillow==9.5.0 -c pytorch 

# install useful modules
pip install tqdm tensorboardX graphviz

# install pyg
#  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA=cu116
pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-1.13.0+${CUDA}.html
pip install torch-sparse==0.6.17 -f https://pytorch-geometric.com/whl/torch-1.13.0+${CUDA}.html
pip install torch-spline-conv==1.2.2 -f https://pytorch-geometric.com/whl/torch-1.13.0+${CUDA}.html
pip install torch-cluster==1.6.1 -f https://pytorch-geometric.com/whl/torch-1.13.0+${CUDA}.html
pip install torch-geometric==1.7.2
pip install requests
pip install pytest-runner
pip install ogb
