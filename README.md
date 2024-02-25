## Prepare environment
pip install torch==1.9.1
datasets are located in the directory ./benchmark, e.g. ./benchmark/MNIST

## Run active learning evaluation
## Mnist
cp config_mnist.py config.py; python -u main.py -d mnist -m NoiseStability -k 30

## Cifar10
cp config_cifar10.py config.py; python -u main.py -d cifar10 -m NoiseStability -k 30
