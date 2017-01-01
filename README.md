# Stacked Generative Adversarial Networks


This repository contains code for the paper "[Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.04357)". Part of the code is modified from OpenAI's [implementation](https://github.com/openai/improved-gan) of Improved GAN.

## Architecture
<p align="center">
<img src="http://www.cs.cornell.edu/~xhuang/img/sgan.jpg" width="650">
</p>

## Samples

<p align="center">
<img src="http://www.cs.cornell.edu/~xhuang/img/mnist_samples.png"  width="250">
<img src="http://www.cs.cornell.edu/~xhuang/img/svhn_samples.png"  width="250">
<img src="http://www.cs.cornell.edu/~xhuang/img/cifar_samples.png"  width="250">
</p>

## Performance Comprison on CIFAR-10
| Method       |  Inception Score | 
| ------------- | ----------- |
| Infusion training    |  4.62 ± 0.06     | 
| GMAN (best variant)  |  5.34 ± 0.05  | 
| LR-GAN  |  6.11 ± 0.06  | 
| EGAN-Ent-VI  |  7.07 ± 0.10  | 
| Denoising feature matching  |  7.72 ± 0.13 | 
| DCGAN  |  6.58 | 
| SteinGAN |  6.35 | 
| Improved GAN（best variant)  |  8.09 ± 0.07 | 
| AC-GAN |  8.25 ± 0.07 | 
| AC-GAN |  8.25 ± 0.07 | 
| **SGAN (ours)**   |  **8.59 ± 0.12** | 

## Citations

If you use the code in this repository in your paper, please consider citing:

```
@article{huang2016sgan,
  title={Stacked Generative Adversarial Networks},
  author={Huang, Xun and Li, Yixuan and Poursaeed, Omid and Hopcroft, John and Belongie, Serge},
  journal={arXiv},
  year={2016}
}
```
