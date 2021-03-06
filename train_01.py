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
import params


LOGGER = logging.getLogger(__name__)


from tasks.copytask_test import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.abc_01 import abcTaskModelTraining, abcTaskParams

TASKS = {
    'abc-01': (abcTaskModelTraining, abcTaskParams),
    'copy-test': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams)
}


# Default values for program arguments
RANDOM_SEED = 10
REPORT_INTERVAL = 2
CHECKPOINT_INTERVAL = 10


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


def progress_bar(percent, last_loss):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), percent, last_loss), end='')


def save_checkpoint(net, name, args, batch_num, losses, costs, valid_accurs, seq_lengths):
    progress_clean()

    basename = "{}/{}-{}-batch-{}".format(args.checkpoint_path, name, args.seed, batch_num)
    model_fname = basename + ".model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    content = {
        "loss": losses,
        "cost": costs,
        "valid_accurs": valid_accurs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def train_batch(net, criterion, optimizer, X, Y):
    """Trains a single batch."""
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        net(X[i])

    # Read the output (no input given)
    y_out = Variable(torch.zeros(Y.size())).to(params.device)
    for i in range(outp_seq_len):
        y_out[i], _ = net()

    loss = criterion(y_out, Y)
    loss.backward()
    clip_grads(net)
    optimizer.step()

    y_out_binarized = y_out.clone().data.cpu()
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data.cpu()))

    return float(loss.cpu().data.numpy())


def evaluate(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = Variable(torch.zeros(Y.size()))
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result

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
    y_out = Variable(torch.zeros(Y.size())).to(params.device)
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

    return nc/nt

def test_model(model):
    accurs = []
    for batch_num, x, y in model.dataloader_valid:
        accur = test_batch(model.net, x, y)
        accurs.append(accur)
        print("\r testing: [{:.2f}]".format(batch_num/len(model.dataloader_valid)), end='')
    mean_cost = np.array(accurs).mean()

    return mean_cost

def train_model(model, args):

    losses = []
    # costs = []
    seq_lengths = []
    start_ms = get_ms()
    for epoch in range(model.params.epoches):
        for percent, x, y in model.dataloader_train:
            # loss, cost = train_batch(model.net, model.criterion, model.optimizer, x, y)
            loss = train_batch(model.net, model.criterion, model.optimizer, x, y)

            losses += [loss]
            # costs += [cost]
            seq_lengths += [y.size(0)]

            # Update the progress bar
            progress_bar(percent, loss)

        # Report
        if (epoch + 1) % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            # mean_cost = np.array(costs[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / len(model.dataloader_train))
            valid_accur = test_model(model)
            progress_clean()

            # LOGGER.info("Epoch %d Loss: %.6f Cost: %.2f Valid Accuracy: %.2f Time: %d ms/sequence",
            #             epoch, mean_loss, mean_cost, valid_accur, mean_time)
            LOGGER.info("Epoch %d Loss: %.6f Valid Accuracy: %.2f Time: %d ms/sequence",
                        epoch, mean_loss, valid_accur, mean_time)
            start_ms = get_ms()

        # Checkpoint
        # if (args.checkpoint_interval != 0) and ((epoch + 1) % args.checkpoint_interval == 0):
        #     valid_accur = test_model(model)

            # save_checkpoint(model.net, model.params.name, args,
            #                 epoch, losses, valid_accur, seq_lengths)

    LOGGER.info("Done training.")


def init_arguments():
    parser = argparse.ArgumentParser(prog='train_01.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()), default='abc-01',
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
    parser.add_argument('-gpu', type=int, default=0,
                        help='gpu index (if could be used)')

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

def init_device(gpu):
    return torch.device(gpu if torch.cuda.is_available() or gpu==-1 else 'cpu')

def main():
    init_logging()

    # Initialize arguments
    args = init_arguments()

    params.device = init_device(args.gpu)
    print(params.device)

    # Initialize random
    init_seed(args.seed)

    # Initialize the model
    model = init_model(args)

    LOGGER.info("Total number of parameters: %d", model.net.calculate_num_params())
    train_model(model, args)


if __name__ == '__main__':
    main()