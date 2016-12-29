# CIFAR experiments

Our code works with Lasagne (version 0.1).

## Sampling with a pretrained model

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sampling.py
```

## Training

Train the bottom GAN:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_gan0.py
```

After training completes, train the top GAN:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_gan1.py
```

(Optionally) train two GANs jointly:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_joint.py
```
