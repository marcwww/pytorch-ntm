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

SOS=0
EOS=1
PAD=2
w2i={'<SOS>':SOS,'<EOS>':EOS,'<PAD>':PAD}
emb_dim=50
VOCB=20
embeddings=torch.nn.Embedding(num_embeddings=VOCB,embedding_dim=emb_dim)
embeddings.weight.to(params.device)
hid2out_module=torch.nn.Linear(emb_dim,VOCB)
hid2out_module.weight.to(params.device)

def to_idx(w):
    if w in w2i.keys():
        return w2i[w]

    num=len(w2i)
    w2i[w]=num
    return num

def to_tensor(seq):
    # (seq_len)
    return torch.LongTensor([to_idx(w) for w in seq])

def load_data(batch_size,sequence_width,f):
    print(params.device)
    # sequence_width=sequence_width-1

    lines = open(f, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('\t') for l in lines]
    pairs = list(random.sample(pairs, len(pairs)))
    batch_src = []
    batch_tar = []
    res=[]
    for i in range(len(pairs)):
        p = pairs[i]
        seq_src, seq_tar = p[0].split(' '), p[1].split(' ')
        batch_src.append(to_tensor(seq_src+['<EOS>']))

        # due to the delimiter to be appended, the EOS is omitted
        batch_tar.append(to_tensor(['<SOS>']+seq_tar+['<EOS>']))
        # batch_tar.append(to_mtrx(seq_tar+EOS, dim=sequence_width))

        if (i + 1) % batch_size == 0:
            mlen_src = max([seq.shape[0] for seq in batch_src])
            mlen_tar = max([seq.shape[0] for seq in batch_tar])

            padded_src = \
                [F.pad(seq, (0, mlen_src - seq.shape[0]),value=PAD)
                 for seq in batch_src]
            padded_tar = \
                [F.pad(seq, (0, mlen_tar - seq.shape[0]),value=PAD)
                 for seq in batch_tar]

            # inp, outp : (seq_len, bsz)
            seqs_inp = torch.stack(padded_src, dim=1)
            seqs_outp = torch.stack(padded_tar, dim=1)

            batch_src.clear()
            batch_tar.clear()

            res.append(((i + 1.0) / len(lines),
                        seqs_inp.to(params.device),
                        seqs_outp.to(params.device)))

    return res

def dataloader(batch_size,sequence_width,f):

    data=load_data(batch_size,sequence_width,f)
    return data

@attrs
class abcTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=emb_dim, convert=int)
    sequence_min_len = attrib(default=1,convert=int)
    sequence_max_len = attrib(default=20, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)
    ftrain = attrib(default='./data/train_abc-500.txt', convert=str)
    fvalid = attrib(default='./data/valid_abc-500.txt', convert=str)
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
    embs = attrib()
    hid2out = attrib()
    vocb =attrib()

    @vocb.default
    def default_vocb(self):
        return VOCB

    @hid2out.default
    def default_hid2out(self):
        return hid2out_module

    @embs.default
    def default_embs(self):
        return embeddings

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width, self.params.sequence_width,
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
        return nn.CrossEntropyLoss(ignore_index=PAD)

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop([{'params':
                                self.net.parameters()},
                              {'params':
                                embeddings.parameters()},
                              {'params':
                                hid2out_module.parameters()}
                              ],
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
