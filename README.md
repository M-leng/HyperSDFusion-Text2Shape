# HyperSDFusion: Bridging Hierarchical Structures in Language and Geometry for Enhanced 3D Text2Shape Generation
[[`arXiv`](https://arxiv.org/abs/2403.00372)]
[[`Project Page`](https://hypersdfusion.github.io/)]
[[`BibTex`](#citation)]

Code release for the CVPR 2024 paper "HyperSDFusion: Bridging Hierarchical Structures in Language and Geometry for Enhanced 3D Text2Shape Generation".


# How to train HyperSDFusion

## Preparing the data

* ShapeNet
    1. Download the ShapeNetV1 dataset from the [official website](https://www.shapenet.org/). Then, extract the downloaded file and put the extracted folder in the `./data` folder. Here we assume the extracted folder is at `./data/ShapeNet/ShapeNetCore.v1`.
    2. Run the following command for preprocessing the SDF from mesh.
```
mkdir -p data/ShapeNet && cd data/ShapeNet
wget [url for downloading ShapeNetV1]
unzip ShapeNetCore.v1.zip
./launchers/unzip_snet_zipfiles.sh # unzip the zip files
cd preprocess
./launchers/launch_create_sdf_shapenet.sh

* text2shape
    - Run the following command for setting up the text2shape dataset.
```
mkdir -p data/ShapeNet/text2shape
wget http://text2shape.stanford.edu/dataset/captions.tablechair.csv -P data/ShapeNet/text2shape
cd preprocess
./launchers/create_snet-text_splits.sh
```

## Training on Text2Shape for text-guided shape generation
```
# text2shape
./launchers/train_sdfusion_txt2shape.sh

# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:

1. Conference version
```BibTeX
come soon
```
2. Arxiv version
Leng Z, Birdal T, Liang X, et al. HyperSDFusion: Bridging Hierarchical Structures in Language and Geometry for Enhanced 3D Text2Shape Generation[J]. arXiv preprint arXiv:2403.00372, 2024.

```BibTeX
@article{leng2024hypersdfusion,
title={HyperSDFusion: Bridging Hierarchical Structures in Language and Geometry for Enhanced 3D Text2Shape Generation},
author={Leng, Zhiying and Birdal, Tolga and Liang, Xiaohui and Tombari, Federico},
journal={arXiv preprint arXiv:2403.00372},
year={2024}
}
```

# Acknowledgement
This code borrows heavily from [SDFusion](https://github.com/yccyenchicheng/SDFusion), [MERU](https://github.com/facebookresearch/meru). We thank the authors for their great work.
This work was done when the author was at TUM.
