"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import numpy as np
import os
import params

from ntm.aio import EncapsulatedNTM

EOS='<EOS>'
w2vec={}
def to_vec(w,dim):
    if w in w2vec.keys():
        return w2vec[w]

    num=len(w2vec)
    vec=torch.zeros(dim)
    vec[num]=1
    w2vec[w]=vec

    return vec

def to_mtrx(seq,dim):
    # mtrx: (seq_len,dim)
    mtrx=torch.stack([to_vec(w,dim) for w in seq])
    return mtrx

def load_data(batch_size,sequence_width,f):
    print(params.device)
    # sequence_width=sequence_width-1

    lines = open(f, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('\t') for l in lines]
    pairs = list(random.sample(pairs, len(pairs)))
    batch_src = []
    batch_tar = []
    for i in range(len(pairs)):
        p = pairs[i]
        seq_src, seq_tar = p[0], p[1]
        batch_src.append(to_mtrx(seq_src, dim=sequence_width))

        # due to the delimiter to be appended, the EOS is omitted
        batch_tar.append(to_mtrx(seq_tar, dim=sequence_width))
        # batch_tar.append(to_mtrx(seq_tar+EOS, dim=sequence_width))

        if (i + 1) % batch_size == 0:
            mlen_src = max([seq.shape[0] for seq in batch_src])
            mlen_tar = max([seq.shape[0] for seq in batch_tar])

            padded_src = \
                [F.pad(seq, (0, 0, 0, mlen_src - seq.shape[0])) for seq in batch_src]
            padded_tar = \
                [F.pad(seq, (0, 0, 0, mlen_tar - seq.shape[0])) for seq in batch_tar]

            # inp, outp : (seq_len, bsz, dim)
            seqs_inp = torch.stack(padded_src, dim=1)
            seqs_outp = torch.stack(padded_tar, dim=1)
            inp = Variable(torch.zeros(mlen_src + 1, batch_size, sequence_width + 1))
            outp = Variable(torch.zeros(mlen_tar + 1, batch_size, sequence_width + 1))
            inp[:mlen_src, :, :sequence_width] = seqs_inp
            outp[:mlen_tar, :, :sequence_width] = seqs_outp

            # delimiter
            inp[mlen_src, :, sequence_width] = 1.0
            outp[mlen_tar, :, sequence_width] = 1.0

            batch_src.clear()
            batch_tar.clear()

            yield (i + 1.0) / len(lines), \
                  inp.to(params.device), \
                  outp.to(params.device)

def dataloader(batch_size,sequence_width,f):

    data=list(load_data(batch_size,sequence_width,f))
    return data

@attrs
class abcTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=12, convert=int)
    sequence_min_len = attrib(default=1,convert=int)
    sequence_max_len = attrib(default=20, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)
    ftrain = attrib(default='./data/train_abc-1000.txt', convert=str)
    fvalid = attrib(default='./data/valid_abc-1000.txt', convert=str)
    epoches = attrib(default=1000, convert=int)
#
# To create a network simply instantiate the `:class:CopyTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = CopyTaskParams(batch_size=4)
# > model = CopyTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:CopyTaskNTM` directly.
#

@attrs
class abcTaskModelTraining(object):
    params = attrib(default=Factory(abcTaskParams))
    net = attrib()
    criterion = attrib()
    optimizer = attrib()
    dataloader_train = attrib()
    dataloader_valid = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width + 1, self.params.sequence_width + 1,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)\
                                .to(params.device)
        return net

    @dataloader_train.default
    def default_dataloader_train(self):
        # device = torch.device(self.params.gpu if torch.cuda.is_available() else "cpu")
        data = load_data(self.params.batch_size,
                         self.params.sequence_width,
                         self.params.ftrain)
        return data

    @dataloader_valid.default
    def default_dataloader_valid(self):
        # device = torch.device(self.params.gpu if torch.cuda.is_available() else "cpu")
        data = load_data(self.params.batch_size,
                         self.params.sequence_width,
                         self.params.fvalid)
        return data

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
