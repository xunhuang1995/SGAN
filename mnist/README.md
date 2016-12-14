# MNIST experiments

Our code works with Lasagne (version 0.1). To run a simple experiment on gpu:

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_independent.py
'''

or on cpu:

'''
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python train_independent.py
'''

This code is still being developed. Currently we only have code for independent training. Joint training code will be available very soon.
