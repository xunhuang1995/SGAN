# MNIST experiments

Our code works with Lasagne (version 0.1). To run a simple experiment with independet training only:

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_independent.py
'''

or run a experiment with joint training:

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_joint.py
'''

This code is still being developed. In the current joint training code all models are trained from scratch. This works OK for MNIST, but for SVHN/CIFAR-10 models need to be initialized with independently pre-trained weights.
