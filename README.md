## [Deep Active Learning with Noise Stability (AAAI 2024)](https://arxiv.org/pdf/2205.13340.pdf)
### Xingjian Li, Pengkun Yang, Yangcheng Gu, Xueying Zhan, Tianyang Wang, Min Xu1, Chengzhong Xu

Uncertainty estimation for unlabeled data is crucial to active
learning. With a deep neural network employed as the backbone model, the data selection process is highly challenging
due to the potential over-confidence of the model inference.
Existing methods resort to special learning fashions (e.g. adversarial) or auxiliary models to address this challenge. This tends
to result in complex and inefficient pipelines, which would
render the methods impractical. In this work, we propose a
novel algorithm that leverages noise stability to estimate data
uncertainty. The key idea is to measure the output derivation
from the original observation when the model parameters are
randomly perturbed by noise. We provide theoretical analyses
by leveraging the small Gaussian noise theory and demonstrate that our method favors a subset with large and diverse
gradients. Our method is generally applicable in various tasks,
including computer vision, natural language processing, and
structural data analysis. It achieves competitive performance
compared against state-of-the-art active learning baselines.

Table of Contents
=================

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Run](#run)
    * [MNIST](#mnist)
    * [CIFAR-10](#cifar-10)
   * [Reference](#reference)

## Setup and Dependencies

1. Create and activate a Conda environment as follows:
```
conda create DALNS
conda activate DALNS
```
2. Install dependencies:
```
pip install torch==1.9.1
```

## Run 
datasets are located in the directory ./benchmark, e.g. ./benchmark/MNIST

### MNIST
1. First copy the file:
```
cp config_mnist.py config.py
```
2. Now, run the following command:
```
python -u main.py -d mnist -m NoiseStability -k 30
```

### CIFAR-10
1. First copy the file:
```
cp config_cifar10.py config.py
```
2. Now, run the following command:
```
python -u main.py -d cifar10 -m NoiseStability -k 30
```

## Reference
If you find this codebase useful in your research, please consider citing our paper:

[insert citation here]
