import sys
import os
import shutil
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
from lasagne.layers import dnn
from lasagne.init import Normal
sys.path.insert(0, '../')
from cifar10_data import unpickle, load_cifar_data
import time
import nn
import scipy
import scipy.misc
from theano.sandbox.rng_mrg import MRG_RandomStreams

''' settings '''
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/sgan_joint')
parser.add_argument('--data_dir', type=str, default='data/cifar-10-python')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--num_epoch', type = int, default = 200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--advloss_weight', type=float, default=1.) # weight for adversarial loss
parser.add_argument('--condloss_weight', type=float, default=1.) # weight for conditional loss
parser.add_argument('--entloss_weight', type=float, default=10.) # weight for entropy loss
parser.add_argument('--labloss_weight', type=float, default=1.) # weight for entropy loss
parser.add_argument('--gen_lr', type=float, default=0.0001) # learning rate for generator
parser.add_argument('--disc_lr', type=float, default=0.0001) # learning rate for discriminator
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()
print(args)

rng = np.random.RandomState(args.seed) # fixed random seeds
theano_rng = MRG_RandomStreams(rng.randint(2 ** 19))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 9)))
data_rng = np.random.RandomState(args.seed_data)


''' input tensor variables '''
y_1hot = T.matrix()
x = T.tensor4()
meanx = T.tensor3()

''' specify generator G1, gen_fc3 = G0(z1, y) '''
z1 = theano_rng.uniform(size=(args.batch_size, 50)) 
gen1_layers = [nn.batch_norm(LL.DenseLayer(LL.InputLayer(shape=(args.batch_size, 50), input_var=z1),
                                           num_units=256, W=Normal(0.02), nonlinearity=T.nnet.relu))] # Input layer for z1
gen1_layer_z = gen1_layers[-1] 

gen1_layers.append(nn.batch_norm(LL.DenseLayer(LL.InputLayer(shape=(args.batch_size, 10), input_var=y_1hot),
                                               num_units=512, W=Normal(0.02), nonlinearity=T.nnet.relu))) # Input layer for labels
gen1_layer_y = gen1_layers[-1]

gen1_layers.append(LL.ConcatLayer([gen1_layer_z,gen1_layer_y],axis=1))
gen1_layers.append(nn.batch_norm(LL.DenseLayer(gen1_layers[-1], num_units=512, W=Normal(0.02), nonlinearity=T.nnet.relu)))
gen1_layers.append(nn.batch_norm(LL.DenseLayer(gen1_layers[-1], num_units=512, W=Normal(0.02), nonlinearity=T.nnet.relu))) 
gen1_layers.append(LL.DenseLayer(gen1_layers[-1], num_units=256, W=Normal(0.02), nonlinearity=T.nnet.relu)) 
                   
''' specify generator G0, gen_x = G0(z0, h1) '''
z0 = theano_rng.uniform(size=(args.batch_size, 16)) # uniform noise
gen0_layers = [LL.InputLayer(shape=(args.batch_size, 16), input_var=z0)] # Input layer for z0
gen0_layers.append(nn.batch_norm(LL.DenseLayer(nn.batch_norm(LL.DenseLayer(gen0_layers[0], num_units=128, W=Normal(0.02), nonlinearity=nn.relu)),
                  num_units=128, W=Normal(0.02), nonlinearity=nn.relu))) # embedding, 50 -> 128
gen0_layer_z_embed = gen0_layers[-1] 

gen0_layers.append(LL.ConcatLayer([gen1_layers[-1],gen0_layer_z_embed], axis=1)) # concatenate noise and fc3 features
gen0_layers.append(LL.ReshapeLayer(nn.batch_norm(LL.DenseLayer(gen0_layers[-1], num_units=256*5*5, W=Normal(0.02), nonlinearity=T.nnet.relu)),
                 (args.batch_size,256,5,5))) # fc
gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,256,10,10), (5,5), stride=(2, 2), padding = 'half',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,128,14,14), (5,5), stride=(1, 1), padding = 'valid',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv

gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,128,28,28), (5,5), stride=(2, 2), padding = 'half',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
gen0_layers.append(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,3,32,32), (5,5), stride=(1, 1), padding = 'valid',
                 W=Normal(0.02),  nonlinearity=T.nnet.sigmoid)) # deconv

gen_fc3, gen_x_pre = LL.get_output([gen1_layers[-1], gen0_layers[-1]], deterministic=True)
gen_x = gen_x_pre - meanx

weights_toload = np.load('pretrained/generator.npz')
weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
LL.set_all_param_values(gen0_layers[-1], weights_list_toload)

''' define sampling functions '''
samplefun = th.function(inputs=[meanx, y_1hot], outputs=gen_x)   # sample function: generating images by stacking all generators

''' load data '''
meanimg = np.load('data/meanimg.npy')

refy = np.zeros((args.batch_size,), dtype=np.int)
for i in range(args.batch_size):
    refy[i] =  i%10
    refy_1hot = np.zeros((args.batch_size, 10),dtype=np.float32)
    refy_1hot[np.arange(args.batch_size), refy] = 1


''' sample images by stacking all generators '''
imgs = samplefun(meanimg, refy_1hot)
imgs = np.transpose(np.reshape(imgs[:100,], (100, 3, 32, 32)), (0, 2, 3, 1))
imgs = [imgs[i] for i in range(100)]
rows = []
for i in range(10):
    rows.append(np.concatenate(imgs[i::10], 1))
imgs = np.concatenate(rows, 0)
scipy.misc.imsave("cifar_samples.png", imgs)

