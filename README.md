# Stacked Generative Adversarial Networks

This repository contains code for the paper "Stacked Generative Adversarial Networks". Part of the code is modified from OpenAI's [implementation](https://github.com/openai/improved-gan) of Improved GAN.

Currently only the code for MNIST experiments is available. I am working hard to polish the code for SVHN/CIFAR-10 experiments and will push them to this repository as soon as possible.

## Usage

```
cd mnist
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_independent.py
```
or if you would like to run on cpu:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python train_independent.py
```

## Citations

If you use the code in this repository in your paper, please cite:

```
@article{huang2016sgan,
  title={Stacked Generative Adversarial Networks},
  author={Huang, Xun and Li, Yixuan and Poursaeed, Omid and Hopcroft, John and Belongie, Serge},
  journal={arXiv},
  year={2016}
}
```
