#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

import argparse
import json
import logging
import time
import random
import re
import sys

import attr
import argcomplete
import torch
from torch.autograd import Variable
import numpy as np
import utils
import crash_on_ipy
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


from tasks.copytask_test import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams),
}


# Default values for program arguments
RANDOM_SEED = 1000
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000


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


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def test_batch(net, X, Y):
    """Trains a single batch."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        net(X[i])

    # Read the output (no input given)
    y_out = Variable(torch.zeros(Y.size()))
    for i in range(outp_seq_len):
        y_out[i], _ = net()

    y_out_binarized = y_out.clone().data.cpu()
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    # cost = torch.sum(torch.abs(y_out_binarized - Y.data.cpu()))
    nc=0
    nt=0
    for i in range(batch_size):
        cost = torch.sum(torch.abs(y_out_binarized[:, i, :] - Y.data.cpu()[:, i, :]))
        if 0 == cost:
            nc+=1
        nt+=1

    return nc, nt

def test_batch_along_length(net, X, Y):
    """Trains a single batch."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        net(X[i])

    # Read the output (no input given)
    y_out = Variable(torch.zeros(Y.size()))
    for i in range(outp_seq_len):
        y_out[i], _ = net()

    y_out_binarized = y_out.clone().data.cpu()
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    # cost = torch.sum(torch.abs(y_out_binarized - Y.data.cpu()))
    nc = defaultdict(int)
    nt = defaultdict(int)

    for i in range(outp_seq_len):
        for j in range(batch_size):
            # for k in range(Y.shape[-1]):
            cost = torch.sum(torch.abs(y_out_binarized[i, j, :] - Y.data.cpu()[i, j, :]))
                # cost = torch.abs(y_out_binarized[i, j, k] - Y.data.cpu()[i, j, k])
                # if cost == 0:
                #     nc[i] += 1
                # nt[i] += 1

            nc[i] += 8 - cost
            nt[i] += 8

    #
    # nc=0
    # nt=0
    # for i in range(batch_size):
    #     cost = torch.sum(torch.abs(y_out_binarized[:, i, :] - Y.data.cpu()[:, i, :]))
    #     if 0 == cost:
    #         nc+=1
    #     nt+=1
    return nc, nt

def test_model_along_length(model, test_data):

    # test_data = torch.load('test_data.pkl')

    nc = defaultdict(int)
    nt = defaultdict(int)
    accur = []

    accurs = []
    example = []
    printnum = random.sample([i+1 for i in range(len(test_data))], 10)
    for batch_num, x, y in test_data:
        print(batch_num/len(test_data))
        nc_batch, nt_batch = test_batch_along_length(model.net, x, y)

        for l, num in nc_batch.items():
            nc[l] += num

        for l, num in nt_batch.items():
            nt[l] += num

    totals = []
    for l in nt.keys():
        accur.append(nc[l]/nt[l])
        totals.append(nt[l])

    return accur, np.array(totals)/np.array(totals).sum(), nc, nt

def evaluate(net, X, Y, plot_name):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)


    utils.show_img(net.memory.memory.data.squeeze(0).numpy(), 'mem')

    # Feed the sequence + delimiter
    w_enc = {'read':[],'write':[]}
    print('-'*20+'encoding'+'-'*20)
    for i in range(inp_seq_len):
        o, state = net(X[i])
        utils.show_img(net.memory.memory.data.squeeze(0).numpy(), 'mem')
        w_enc['read'].append(state[-1][0])
        w_enc['write'].append(state[-1][1])

    # Read the output (no input given)
    w_dec = {'read': [], 'write': []}
    print('-'*20+'decoding'+'-'*20)
    y_out = Variable(torch.zeros(Y.size()))
    for i in range(outp_seq_len):
        y_out[i], state = net()
        utils.show_img(net.memory.memory.data.squeeze(0).numpy(), 'mem')
        w_dec['read'].append(state[-1][0])
        w_dec['write'].append(state[-1][1])

    w_enc_read = torch.cat(w_enc['read'])
    w_enc_write = torch.cat(w_enc['write'])
    w_dec_read = torch.cat(w_dec['read'])
    w_dec_write = torch.cat(w_dec['write'])

    boundary_enc = torch.Tensor(np.array([[0.5] * w_enc_write.shape[0]]))
    boundary_dec = torch.Tensor(np.array([[0.5] * w_dec_write.shape[0]]))
    # w = torch.cat(
    #     [w_enc_read,
    #      boundary,
    #      w_enc_write,
    #      boundary,
    #      w_dec_read,
    #      boundary,
    #      w_dec_write])
    w_enc = torch.cat(
        [w_enc_read,
         boundary_enc.transpose(0,1),
         w_enc_write], dim=1)

    w_dec =  torch.cat([
         w_dec_read,
        boundary_dec.transpose(0,1),
         w_dec_write], dim=1)

    boundary = torch.Tensor(np.array([[0.5] * 41]))
    w = torch.cat([w_enc, boundary, w_dec])
    #
    utils.show_img(w.data.numpy(), plot_name)
    # utils.show_img(w_enc.data.numpy(), plot_name)
    # utils.show_img(w_dec.data.numpy(), plot_name)

    # utils.show_img(w_enc_read)
    # utils.show_img(w_enc_write)
    # utils.show_img(w_dec_read)
    # utils.show_img(w_dec_write)


    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        # 'states': states
    }

    return result

def test_model(model, test_data):

    # test_data = torch.load('test_data.pkl')

    accurs = []
    example = []
    nc = 0
    nt = 0
    for batch_num, x, y in test_data:
        print(batch_num/len(test_data))
        nc_batch, nt_batch = test_batch(model.net, x, y)
        nc += nc_batch
        nt += nt_batch

    return nc/nt

def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()), default='copy',
                        help="Choose the task to train (default: copy)")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL,
                        help="Checkpoint interval (default: {}). "
                             "Use 0 to disable checkpointing".format(CHECKPOINT_INTERVAL))
    parser.add_argument('--checkpoint-path', action='store', default='./',
                        help="Path for saving checkpoint data (default: './')")
    parser.add_argument('--report-interval', type=int, default=REPORT_INTERVAL,
                        help="Reporting interval")

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path.rstrip('/')

    return args


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params

def init_model(args):
    LOGGER.info("Training for the **%s** task", args.task)

    model_cls, params_cls = TASKS[args.task]
    params = params_cls()
    params = update_model_params(params, args.param)

    LOGGER.info(params)

    model = model_cls(params=params)
    return model


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def main():
    init_logging()

    # Initialize arguments
    args = init_arguments()

    # Initialize random
    init_seed(args.seed)

    # Initialize the model
    model = init_model(args)

    # load_path = 'train_test_mid/copy-task-1000-batch-50000.model'
    # load_path = 'result-train5-n20-mirror/mirror-task-1000-batch-50000.model'
    # load_path = 'result-train5-test10-n20/copy-task-1000-batch-50000.model'
    # load_path = 'result-train5-n20/copy-task-1000-batch-50000.model'
    # load_path = '1457mem10bsz4-1000-batch-13000.model'
    # load_path = 'res-1457-mirror/mirror1457mem20bsz2-1000-batch-46000.model'
    load_path = 'train_test_end-10-batch-7000.model'

    model.net.load_state_dict(torch.load(load_path))

    test_data = model.dataloader_valid
    # cost, _ = test_model(model, test_data)
    # print('accur:', cost)
    # print(test_model_along_length(model, test_data))
    print(test_model(model, test_data))

if __name__ == '__main__':
    main()
