import numpy as np
from torch.nn.init import xavier_uniform_
from collections import defaultdict
import time
import logging
LOGGER = logging.getLogger(__name__)
import torch
import random
import matplotlib.pyplot as plt

def draw_dist(data, bins):
    plt.hist(data, bins=bins)
    plt.show()

def show_img(img, plot_name):
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.imsave(plot_name, img)

def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.requires_grad:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def balance_bias(train_iter):

    nlbls = defaultdict(int)
    for batch in train_iter:
        for lbl in batch.lbl.squeeze(0):
            nlbls[lbl.item()] += 1

    res = []
    for i in range(len(nlbls)):
        res.append(1/nlbls[i])

    res = np.array(res)
    res /= res.sum()

    return res

def shift_matrix(n):
    W_up = np.eye(n)
    for i in range(n-1):
        W_up[i,:] = W_up[i+1,:]
    W_up[n-1,:] *= 0
    W_down = np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:] = W_down[i-1,:]
    W_down[0,:] *= 0
    return W_up,W_down

def avg_vector(i, n):
    V = np.zeros(n)
    V[:i+1] = 1/(i+1)
    return V

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            xavier_uniform_(p)

def filter_state_dict(state_dict):
    keys = []
    for k in state_dict.keys():
        if not state_dict[k].requires_grad:
            keys.append(k)
    for k in keys:
        state_dict.pop(k)

    return state_dict

def progress_bar(percent, last_loss, epoch):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f}/epoch {:d} (Loss: {:.4f} )".format(
        "=" * fill,
        " " * (40 - fill),
        percent,
        epoch,
        last_loss),
        end='')

def select_1d(data, index, device):
    # data: (N, M)
    # index: (N)
    N, M = data.shape
    idx = index + torch.LongTensor(range(N)).to(device) * M
    # res: (N)
    res = data.view(-1)[idx.view(-1)]
    return res

if __name__ == '__main__':
    draw_dist([1,2,3,1,2,3,2,1], 10)