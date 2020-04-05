from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import argparse


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env',
                        help='The gym environment to train on',
                        type=str,
                        required=True)
    parser.add_argument('--model-path',
                        help='The path to the save the pytorch model',
                        type=str,
                        required=False)
    parser.add_argument('--gamma',
                        help='Gamma parameter',
                        type=float,
                        default=0.99,
                        required=False)
    parser.add_argument('--output-path',
                        help='The output directory to store training stats',
                        type=str,
                        required=False)
    parser.add_argument('--gpu',
                        help='Use the gpu or not',
                        action='store_true',
                        required=False)
    parser.add_argument('--render',
                        help='Render visual or not',
                        action='store_true',
                        required=False)
    parser.add_argument('--render-episodes',
                        help='Render every these many episodes',
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument('--episodes',
                        help='Number of episodes to run for',
                        type=int,
                        default=2000,
                        required=False)
    parser.add_argument('--batchsize',
                        help='Number of experiences sampled from replay buffer',
                        type=int,
                        default=256,
                        required=False)
    parser.add_argument('--update-steps',
                        help='Number of steps to update for per episode',
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument('--gradient-clip',
                        help='How much to clip the gradients by',
                        type=float,
                        default=2.5,
                        required=False)
    parser.add_argument('--reward-clip',
                        help='How much to clip reward, clipped in [-rc, rc]',
                        type=float,
                        default=20,
                        required=False)
    parser.add_argument('--epsilon-decay',
                        help='Parameter for epsilon decay exploration',
                        type=float,
                        default=2.75,
                        required=False)
    parser.add_argument('--epsilon-decay-start',
                        help='After this episode, epsilon decay starts',
                        type=float,
                        default=5,
                        required=False)
    parser.add_argument('--replay-buffer-size',
                        help='Max size of replay buffer',
                        type=int,
                        default=500000,
                        required=False)
    parser.add_argument('--lr',
                        help='Learning rate for the optimizer',
                        type=float,
                        default=5e-4,
                        required=False)
    parser.add_argument('--target-moving-average',
                        help='EMA parameter for target network',
                        type=float,
                        default=5e-3,
                        required=False)
    parser.add_argument('--seed',
                        help='The random seed for this run',
                        type=int,
                        default=10,
                        required=False)

    return parser.parse_args()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        torch.nn.init.uniform_(m.bias, -1, 1)


def sync_networks(target, online, alpha):
    for online_param, target_param in zip(online.parameters(),
                                          target.parameters()):
        target_param.data.copy_(alpha * online_param.data +
                                (1 - alpha) * target_param.data)


# Thanks to RoshanRane - Pytorch forums
# (https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)
# Dec 2018
def plot_grad_flow(named_parameters, ep=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)
    ], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return plt.gcf()
