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
import time
sys.path.insert(0, '../')
import nn
import scipy
import scipy.misc
from theano.sandbox.rng_mrg import MRG_RandomStreams

''' settings '''
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='logs/test')
parser.add_argument('--save_interval', type = int, default = 1)
parser.add_argument('--num_epoch', type = int, default = 200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--advloss_weight', type=float, default=1.) # weight for adversarial loss
parser.add_argument('--condloss_weight', type=float, default=1.) # weight for conditional loss
parser.add_argument('--entloss_weight', type=float, default=10.) # weight for entropy loss
parser.add_argument('--gen_lr', type=float, default=0.0001) # learning rate for generator
parser.add_argument('--disc_lr', type=float, default=0.0001) # learning rate for discriminator
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()
print(args)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir) # make out_dir if it does not exist, copy current script to out_dir
    print "Created folder {}".format(args.out_dir)
    shutil.copyfile(sys.argv[0], args.out_dir + '/training_script.py')
else:
    print "folder {} already exists. please remove it first.".format(args.out_dir)
    exit(1)

rng = np.random.RandomState(args.seed) # fixed random seeds
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

''' specify pre-trained encoder E '''
enc_layers = [LL.InputLayer(shape=(None, 1, 28, 28), input_var=None)]
enc_layer_conv1 = dnn.Conv2DDNNLayer(enc_layers[-1], 32, (5,5), pad=0, stride=1, W=Normal(0.01), nonlinearity=nn.relu)
enc_layers.append(enc_layer_conv1)
enc_layer_pool1 = LL.MaxPool2DLayer(enc_layers[-1], pool_size=(2, 2))
enc_layers.append(enc_layer_pool1)
enc_layer_conv2 = dnn.Conv2DDNNLayer(enc_layers[-1], 32, (5,5), pad=0, stride=1, W=Normal(0.01), nonlinearity=nn.relu)
enc_layers.append(enc_layer_conv2)
enc_layer_pool2 = LL.MaxPool2DLayer(enc_layers[-1], pool_size=(2, 2))
enc_layers.append(enc_layer_pool2)
enc_layer_fc3 = LL.DenseLayer(enc_layers[-1], num_units=256, nonlinearity=T.nnet.relu)
enc_layers.append(enc_layer_fc3)
enc_layer_fc4 = LL.DenseLayer(enc_layers[-1], num_units=10, nonlinearity=T.nnet.softmax)
enc_layers.append(enc_layer_fc4)

''' load pretrained weights for encoder '''
weights_toload = np.load('pretrained/encoder.npz')
weights_list_toload = [weights_toload['arr_{}'.format(k)] for k in range(len(weights_toload.files))]
LL.set_all_param_values(enc_layers[-1], weights_list_toload)

''' input tensor variables '''
y_1hot = T.matrix()
x = T.tensor4()
lr = T.scalar() # learning rate
real_fc3 = LL.get_output(enc_layer_fc3, x, deterministic=True)

''' specify generator G1, gen_fc3 = G0(z1, y) '''
z1 = theano_rng.uniform(size=(args.batch_size, 50))
gen1_layers = [LL.InputLayer(shape=(args.batch_size, 50), input_var=z1)] # Input layer for z1
gen1_layer_z = gen1_layers[-1]

gen1_layers.append(LL.InputLayer(shape=(args.batch_size, 10), input_var=y_1hot)) # Input layer for labels
gen1_layer_y = gen1_layers[-1]

gen1_layers.append(LL.ConcatLayer([gen1_layer_z,gen1_layer_y],axis=1))
gen1_layers.append(nn.batch_norm(LL.DenseLayer(gen1_layers[-1], num_units=512, W=Normal(0.02), nonlinearity=T.nnet.relu)))
gen1_layers.append(nn.batch_norm(LL.DenseLayer(gen1_layers[-1], num_units=512, W=Normal(0.02), nonlinearity=T.nnet.relu)))
gen1_layers.append(LL.DenseLayer(gen1_layers[-1], num_units=256, W=Normal(0.02), nonlinearity=T.nnet.relu))

gen_fc3 = LL.get_output(gen1_layers[-1], deterministic=False)

''' specify generator G0, gen_x = G0(z0, h1) '''
z0 = theano_rng.uniform(size=(args.batch_size, 50)) # uniform noise
gen0_layers = [LL.InputLayer(shape=(args.batch_size, 50), input_var=z0)] # Input layer for z0
gen0_layers.append(nn.batch_norm(LL.DenseLayer(gen0_layers[-1], num_units=128, W=Normal(0.02)))) # embedding, 50 -> 128
gen0_layer_z_embed = gen0_layers[-1]

gen0_layers.append(LL.InputLayer(shape=(args.batch_size, 256), input_var=real_fc3)) # Input layer for real_fc3 in independent training, gen_fc3 in joint training
gen0_layer_fc3 = gen0_layers[-1]

gen0_layers.append(LL.ConcatLayer([gen0_layer_fc3,gen0_layer_z_embed], axis=1)) # concatenate noise and fc3 features
gen0_layers.append(LL.ReshapeLayer(nn.batch_norm(LL.DenseLayer(gen0_layers[-1], num_units=128*4*4, W=Normal(0.02), nonlinearity=T.nnet.relu)),
                 (args.batch_size,128,4,4))) # fc
gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,128,8,8), (5,5), stride=(2, 2), padding = 'half',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,64,12,12), (5,5), stride=(1, 1), padding = 'valid',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv

gen0_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,64,24,24), (5,5), stride=(2, 2), padding = 'half',
                 W=Normal(0.02),  nonlinearity=nn.relu))) # deconv
gen0_layers.append(nn.Deconv2DLayer(gen0_layers[-1], (args.batch_size,1,28,28), (5,5), stride=(1, 1), padding = 'valid',
                 W=Normal(0.02),  nonlinearity=T.nnet.sigmoid)) # deconv

gen_x = LL.get_output(gen0_layers[-1], deterministic=False)
gen_x_joint = LL.get_output(gen0_layers[-1], {gen0_layer_fc3: gen_fc3}, deterministic=False)

''' specify discriminator D1 '''
disc1_layers = [LL.InputLayer(shape=(None, 256))]
disc1_layers.append(LL.DenseLayer(disc1_layers[-1], num_units=256))
disc1_layers.append(LL.DenseLayer(disc1_layers[-1], num_units=256))
disc1_layer_shared = disc1_layers[-1]

disc1_layer_z_recon = LL.DenseLayer(disc1_layer_shared, num_units=50, nonlinearity=T.nnet.sigmoid)
disc1_layers.append(disc1_layer_z_recon)

disc1_layer_adv = LL.DenseLayer(disc1_layer_shared, num_units=1, nonlinearity=T.nnet.sigmoid)
disc1_layers.append(disc1_layer_adv)

''' specify discriminator D0 '''
disc0_layers = [LL.InputLayer(shape=(args.batch_size, 28**2))]
disc0_layers.append(LL.ReshapeLayer(disc0_layers[-1], (args.batch_size, 1, 28, 28)))
disc0_layers.append(dnn.Conv2DDNNLayer(disc0_layers[-1], 32, (5,5), pad=2, stride=2, W=Normal(0.01), nonlinearity=nn.lrelu)) # conv
disc0_layers.append(dnn.Conv2DDNNLayer(disc0_layers[-1], 64, (5,5), pad=2, stride=2, W=Normal(0.01), nonlinearity=nn.lrelu)) # conv
disc0_layers.append(dnn.Conv2DDNNLayer(disc0_layers[-1], 128, (5,5), pad=2, stride=2, W=Normal(0.01), nonlinearity=nn.lrelu)) # conv
disc0_layers.append(LL.DenseLayer(disc0_layers[-1], num_units=256, nonlinearity=T.nnet.relu)) # fc
disc0_layer_shared = disc0_layers[-1]

disc0_layer_z_recon = LL.DenseLayer(disc0_layer_shared, num_units=50, nonlinearity=T.nnet.sigmoid) # branch for Q0, trained to recover noise
disc0_layers.append(disc0_layer_z_recon)

disc0_layer_adv = LL.DenseLayer(disc0_layer_shared, num_units=1, nonlinearity=T.nnet.sigmoid) # branch for D0, trained to classify fake v.s. real
disc0_layers.append(disc0_layer_adv)

''' forward pass '''
prob_real1 = LL.get_output(disc1_layer_adv, real_fc3, deterministic=False)
prob_gen1, recon_z1 = LL.get_output([disc1_layer_adv, disc1_layer_z_recon], gen_fc3, deterministic=False)
prob_real0 = LL.get_output(disc0_layer_adv, x, deterministic=False)
prob_gen0, recon_z0 = LL.get_output([disc0_layer_adv, disc0_layer_z_recon], gen_x_joint, deterministic=False)

''' loss for discriminator and Q '''
loss_real1 = T.mean(T.nnet.binary_crossentropy(prob_real1, T.ones(prob_real1.shape)))
loss_fake1 = T.mean(T.nnet.binary_crossentropy(prob_gen1, T.zeros(prob_gen1.shape)))
loss_gen1_ent = T.mean((recon_z1 - z1)**2)
loss_disc1 = args.advloss_weight * (0.5*loss_real1  + 0.5*loss_fake1) + args.entloss_weight * loss_gen1_ent

loss_real0 = T.mean(T.nnet.binary_crossentropy(prob_real0, T.ones(prob_real0.shape)))
loss_fake0 = T.mean(T.nnet.binary_crossentropy(prob_gen0, T.zeros(prob_gen0.shape)))
loss_gen0_ent = T.mean((recon_z0 - z0)**2)
loss_disc0 = args.advloss_weight * (0.5*loss_real0  + 0.5*loss_fake0) + args.entloss_weight * loss_gen0_ent

''' loss for generator '''
recon_y = LL.get_output(enc_layer_fc4, {enc_layer_fc3: gen_fc3}) # reconstructed labels
loss_gen1_adv = T.mean(T.nnet.binary_crossentropy(prob_gen1, T.ones(prob_gen1.shape))) # adversarial loss
loss_gen1_cond = T.mean(T.nnet.categorical_crossentropy(recon_y, y_1hot)) # conditional loss, cross entropy for labels
loss_gen1 = args.advloss_weight * loss_gen1_adv + args.condloss_weight * loss_gen1_cond + args.entloss_weight * loss_gen1_ent

recon_fc3 = LL.get_output(enc_layer_fc3, gen_x_joint) # reconstructed fc3 activations
loss_gen0_adv = T.mean(T.nnet.binary_crossentropy(prob_gen0, T.ones(prob_gen0.shape))) # adversarial loss
loss_gen0_cond = T.mean((recon_fc3 - gen_fc3)**2) # conditional loss, euclidean distance in feature space
loss_gen0 = args.advloss_weight * loss_gen0_adv + args.condloss_weight * loss_gen0_cond + args.entloss_weight * loss_gen0_ent

''' collect parameter updates for discriminators '''
disc1_params = LL.get_all_params(disc1_layers, trainable=True)
disc1_param_updates = nn.adam_updates(disc1_params, loss_disc1, lr=lr, mom1=0.5)
disc1_bn_updates = [u for l in LL.get_all_layers(disc1_layers[-1]) for u in getattr(l,'bn_updates',[])]
disc1_bn_params = []
for l in LL.get_all_layers(disc1_layers[-1]):
    if hasattr(l, 'avg_batch_mean'):
        disc1_bn_params.append(l.avg_batch_mean)
        disc1_bn_params.append(l.avg_batch_var)

disc0_params = LL.get_all_params(disc0_layers, trainable=True)
disc0_param_updates = nn.adam_updates(disc0_params, loss_disc0, lr=lr, mom1=0.5)
disc0_bn_updates = [u for l in LL.get_all_layers(disc0_layers[-1]) for u in getattr(l,'bn_updates',[])]
disc0_bn_params = []
for l in LL.get_all_layers(disc0_layers[-1]):
    if hasattr(l, 'avg_batch_mean'):
        disc0_bn_params.append(l.avg_batch_mean)
        disc0_bn_params.append(l.avg_batch_var)

''' collect parameter updates for generators '''
gen1_params = LL.get_all_params(gen1_layers[-1], trainable=True)
gen1_param_updates = nn.adam_updates(gen1_params, loss_gen1, lr=lr, mom1=0.5)
gen1_bn_updates = [u for l in LL.get_all_layers(gen1_layers[-1]) for u in getattr(l,'bn_updates',[])]
gen1_bn_params = []
for l in LL.get_all_layers(gen1_layers[-1]):
    if hasattr(l, 'avg_batch_mean'):
        gen1_bn_params.append(l.avg_batch_mean)
        gen1_bn_params.append(l.avg_batch_var)

gen0_params = LL.get_all_params(gen0_layers[-1], trainable=True)
gen0_param_updates = nn.adam_updates(gen0_params, loss_gen0, lr=lr, mom1=0.5)
gen0_bn_updates = [u for l in LL.get_all_layers(gen0_layers[-1]) for u in getattr(l,'bn_updates',[])]
gen0_bn_params = []
for l in LL.get_all_layers(gen0_layers[-1]):
    if hasattr(l, 'avg_batch_mean'):
        gen0_bn_params.append(l.avg_batch_mean)
        gen0_bn_params.append(l.avg_batch_var)

''' define training and testing functions '''
train_batch_disc = th.function(inputs=[x, y_1hot, lr], outputs=[gen_fc3, real_fc3, gen_x_joint, x],
    updates=disc1_param_updates+disc1_bn_updates+disc0_param_updates+disc0_bn_updates)
train_batch_gen = th.function(inputs=[y_1hot, lr], outputs=[loss_gen1_adv, loss_gen1_cond, loss_gen1_ent, loss_gen0_adv, loss_gen0_cond, loss_gen0_ent],
    updates=gen1_param_updates+gen1_bn_updates+gen0_param_updates+gen0_bn_updates)
samplefun = th.function(inputs=[y_1hot], outputs=gen_x_joint)   # sample function: generating images by stacking all generators
reconfun = th.function(inputs=[x], outputs=gen_x)       # reconstruction function: use the bottom generator
                                                        # to generate images conditioned on real fc3 features
''' load data '''
data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainx = np.reshape(trainx, (trainx.shape[0],1,28,28))
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0]/args.batch_size)
# testx = data['x_test'].astype(th.config.floatX)
# testy = data['y_test'].astype(np.int32)
# nr_batches_test = int(testx.shape[0]/args.batch_size)

refy = np.zeros((args.batch_size,), dtype=np.int)
for i in range(args.batch_size):
    refy[i] =  i%10
    refy_1hot = np.zeros((args.batch_size, 10),dtype=np.float32)
    refy_1hot[np.arange(args.batch_size), refy] = 1

''' perform training  '''
logs = {'loss_gen1_adv': [], 'loss_gen1_cond': [], 'loss_gen1_ent': [], 'var_gen1': [], 'var_real1': [],
        'loss_gen0_adv': [], 'loss_gen0_cond': [], 'loss_gen0_ent': [], 'var_gen0': [], 'var_real0': []} # training logs
for epoch in range(args.num_epoch):
    begin = time.time()

    ''' shuffling '''
    inds = rng.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]

    for t in range(nr_batches_train):
        ''' construct minibatch '''
        batchx = trainx[t*args.batch_size:(t+1)*args.batch_size]
        batchy = trainy[t*args.batch_size:(t+1)*args.batch_size]
        batchy_1hot = np.zeros((args.batch_size, 10), dtype=np.float32)
        batchy_1hot[np.arange(args.batch_size), batchy] = 1 # convert to one-hot label
        # randomy = np.random.randint(10, size = (args.batch_size,))
        # randomy_1hot = np.zeros((args.batch_size, 10),dtype=np.float32)
        # randomy_1hot[np.arange(args.batch_size), randomy] = 1

        ''' train discriminators '''
        g1, r1, g0, r0 = train_batch_disc(batchx, batchy_1hot, args.disc_lr)

        ''' train generators '''
        l_gen1_adv, l_gen1_cond, l_gen1_ent, l_gen0_adv, l_gen0_cond, l_gen0_ent = train_batch_gen(batchy_1hot, args.gen_lr)

        ''' store log information '''
        logs['loss_gen1_adv'].append(l_gen1_adv)
        logs['loss_gen1_cond'].append(l_gen1_cond)
        logs['loss_gen1_ent'].append(l_gen1_ent)
        logs['var_gen1'].append(np.var(np.array(g1)))
        logs['var_real1'].append(np.var(np.array(r1)))

        logs['loss_gen0_adv'].append(l_gen0_adv)
        logs['loss_gen0_cond'].append(l_gen0_cond)
        logs['loss_gen0_ent'].append(l_gen0_ent)
        logs['var_gen0'].append(np.var(np.array(g0)))
        logs['var_real0'].append(np.var(np.array(r0)))

    ''' sample images by stacking all generators'''
    imgs = samplefun(refy_1hot)
    imgs = np.reshape(imgs[:100,], (100, 28, 28))
    imgs = [imgs[i, :, :] for i in range(100)]
    rows = []
    for i in range(10):
        rows.append(np.concatenate(imgs[i::10], 1))
    imgs = np.concatenate(rows, 0)
    scipy.misc.imsave(args.out_dir + "/mnist_sample_epoch{}.png".format(epoch), imgs)

    ''' original images in the training set'''
    orix = np.reshape(batchx[:100,], (100, 28, 28))
    orix = [orix[i, :, :] for i in range(100)]
    rows = []
    for i in range(10):
        rows.append(np.concatenate(orix[i::10], 1))
    orix = np.concatenate(rows, 0)
    scipy.misc.imsave(args.out_dir + "/mnist_ori_epoch{}.png".format(epoch), orix)

    ''' reconstruct images '''
    reconx = reconfun(batchx)
    reconx = np.reshape(reconx[:100,], (100, 28, 28))
    reconx = [reconx[i, :, :] for i in range(100)]
    rows = []
    for i in range(10):
        rows.append(np.concatenate(reconx[i::10], 1))
    reconx = np.concatenate(rows, 0)
    scipy.misc.imsave(args.out_dir + "/mnist_recon_epoch{}.png".format(epoch), reconx)

    ''' save params '''
    if epoch%args.save_interval==0:
        np.savez(args.out_dir + "/disc1_params_epoch{}.npz".format(epoch), *LL.get_all_param_values(disc1_layers[-1]))
        np.savez(args.out_dir + '/gen1_params_epoch{}.npz'.format(epoch), *LL.get_all_param_values(gen1_layers[-1]))
        np.savez(args.out_dir + "/disc0_params_epoch{}.npz".format(epoch), *LL.get_all_param_values(disc0_layers[-1]))
        np.savez(args.out_dir + '/gen0_params_epoch{}.npz'.format(epoch), *LL.get_all_param_values(gen0_layers[-1]))
        np.save(args.out_dir + '/logs.npy',logs)


    print("Epoch %d, time = %ds, var gen fc3 = %.4f, var real fc3 = %.4f, var gen x = %.4f, var real x = %.4f" %
         (epoch, time.time()-begin, np.var(np.array(g1)), np.var(np.array(r1)), np.var(np.array(g0)), np.var(np.array(r0))))
    print("loss_gen0_adv = %.4f,  loss_gen0_cond = %.4f, loss_gen0_ent = %.4f"
        % (l_gen0_adv, l_gen0_cond, l_gen0_ent))
    print("loss_gen1_adv = %.4f,  loss_gen1_cond = %.4f, loss_gen1_ent = %.4f"
        % (l_gen1_adv, l_gen1_cond, l_gen1_ent))
