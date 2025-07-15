## Getting Started
our code is basically refered from:

@misc{almuzairee2024recipe,
      title={A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning}, 
      author={Abdulaziz Almuzairee and Nicklas Hansen and Henrik I. Christensen},
      year={2024},
      eprint={2405.17416},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

https://github.com/aalmuzairee/dmcgb2

### Packages

All package dependencies can be installed with the following commands. We assume that you have access to a GPU with CUDA >=11.0 support:

```
conda env create -f environment.yaml
conda activate sada
```
If building from docker, we recommend using `nvidia/cudagl:11.3.0-base-ubuntu18.04` as the base image.

-----

### Datasets

This repository has dependencies on external datasets. For full functionality, you need to download the following datasets:

- Places365 Dataset: For applying Random Overlay Image Augmentation, we follow [SODA](https://github.com/nicklashansen/dmcontrol-generalization-benchmark) in using the [Places365](http://places2.csail.mit.edu/download.html) dataset 
- DAVIS Dataset: For evaluating on the [Distracting Control Suite](https://github.com/google-research/google-research/tree/master/distracting_control), the [DAVIS](https://davischallenge.org/davis2017/code.html) dataset is used for video backgrounds

#### Easy Install

We provide utility scripts for installing these datasets in `scripts` folder, which can be run using 

```
scripts/install_places.sh
scripts/install_davis.sh
```

#### Manual Install 

If you prefer manual installation, the Places365 Dataset can be downloaded by running:

```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

The DAVIS dataset can be downloaded by running:

```
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```

After downloading and extracting the data, add your dataset directory to the `datasets` list in `cfgs/config.yaml`.

-----


further complete soon...