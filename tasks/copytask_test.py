"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import params

from ntm.aio import EncapsulatedNTM

data_train=[]

def gen(batch_size,
        seq_width,
        min_len,
        max_len):
    seq_len = random.randint(min_len, max_len)
    seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
    seq = Variable(torch.from_numpy(seq))

    # The input includes an additional channel used for the delimiter
    inp = Variable(torch.zeros(seq_len + 1, batch_size, seq_width + 1))
    inp[:seq_len, :, :seq_width] = seq
    inp[seq_len, :, seq_width] = 1.0  # delimiter in our control channel
    outp = seq.clone()

    return inp,outp

# Generator of randomized test sequences
def dataloader_train(num_batches,
               batch_size,
               seq_width,
               min_len,
               max_len,
               train_ratio):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(num_batches):

        if batch_num/num_batches < train_ratio:
            # All batches have the same sequence length
            inp,outp=gen(batch_size,seq_width,min_len,max_len)
            data_train.append((inp,outp))
            yield batch_num+1, inp.float().to(params.device), outp.float().to(params.device)

        else:
            inp, outp = data_train[batch_num % len(data_train)]
            yield batch_num+1, inp.float().to(params.device), outp.float().to(params.device)

def dataloader_valid(num_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):

    res=[]
    for batch_num in range(num_batches):
        # All batches have the same sequence length
        inp,outp=gen(batch_size,seq_width,min_len,max_len)

        res.append((batch_num+1, inp.float().to(params.device), outp.float().to(params.device)))

    return res



@attrs
class CopyTaskParams(object):
    name = attrib(default="copy-task-test")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=1, convert=int)
    sequence_min_len = attrib(default=2,convert=int)
    sequence_max_len = attrib(default=3, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_samples_train = attrib(default=1000000, convert=int)
    num_samples_valid = attrib(default=1000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)
    train_ratio = attrib(default=0.1, convert=float)


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
class CopyTaskModelTraining(object):
    params = attrib(default=Factory(CopyTaskParams))
    net = attrib()
    dataloader_train = attrib()
    dataloader_valid = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width + 1, self.params.sequence_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m).to(params.device)
        return net

    @dataloader_train.default
    def default_dataloader_train(self):
        return dataloader_train(int(self.params.num_samples_train/self.params.batch_size), self.params.batch_size,
                          self.params.sequence_width,
                          self.params.sequence_min_len,
                          self.params.sequence_max_len,
                          self.params.train_ratio)

    @dataloader_valid.default
    def default_dataloader_valid(self):
        return dataloader_valid(int(self.params.num_samples_valid/self.params.batch_size),self.params.batch_size,
                               self.params.sequence_width,
                               self.params.sequence_min_len+2,
                               self.params.sequence_max_len+2)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
