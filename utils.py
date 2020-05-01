from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import argparse
from datetime import datetime
from atari_wrappers import AtariPreprocess, MaxAndSkipEnv, FrameStack


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env',
                        help='The gym environment to train on',
                        type=str,
                        required=True)
    parser.add_argument('--model-type',
                        help="Type of architecture",
                        type=str,
                        default='mlp',
                        choices=["cnn", "mlp"],
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
    parser.add_argument('--load-checkpoint-path',
                        help='Use the gpu or not',
                        type=str,
                        required=False)
    parser.add_argument('--no-atari',
                        help='Use atari preprocessing',
                        action='store_false',
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
    parser.add_argument('--num-frames',
                        help='Number of frames to stack (CNN only)',
                        type=int,
                        default=4,
                        required=False)
    parser.add_argument('--max-steps',
                        help='Number of steps to run for',
                        type=int,
                        default=100000,
                        required=False)
    parser.add_argument('--checkpoint-steps',
                        help='Checkpoint every so often',
                        type=int,
                        default=10000,
                        required=False)
    parser.add_argument('--test-policy-episodes',
                        help='Policy is tested every these many episodes',
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument('--warmup-period',
                        help='Number of steps to act randomly and not train',
                        type=int,
                        default=50000,
                        required=False)
    parser.add_argument('--batchsize',
                        help='Number of experiences sampled from replay buffer',
                        type=int,
                        default=256,
                        required=False)
    parser.add_argument('--update-steps',
                        help='Number of steps to update for per episode',
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument('--gradient-clip',
                        help='How much to clip the gradients by',
                        type=float,
                        default=2.5,
                        required=False)
    parser.add_argument('--reward-clip',
                        help='How much to clip reward, clipped in [-rc, rc], 0 \
                        results in unclipped',
                        type=float,
                        default=0,
                        required=False)
    parser.add_argument('--epsilon-decay',
                        help='Parameter for epsilon decay',
                        type=int,
                        default=1e3,
                        required=False)
    parser.add_argument('--epsilon-decay-end',
                        help='Parameter for epsilon decay end',
                        type=int,
                        default=0.1,
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
    parser.add_argument('--vanilla-DQN',
                        help='Use the vanilla dqn update instead of double DQN',
                        action='store_true',
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


# Adapted from pytorch tutorials:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def conv2d_size_out(size, kernel_size, stride):
    return ((size[0] - (kernel_size[0] - 1) - 1) // stride + 1,
            (size[1] - (kernel_size[1] - 1) - 1) // stride + 1)


def deque_to_tensor(last_num_frames):
    """ Convert deque of n frames to tensor """
    return torch.cat(list(last_num_frames), dim=0)


def make_atari(env, num_frames):
    """ Wrap env in atari processed env """
    return FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4),
                      num_frames)


# Thanks to RoshanRane - Pytorch forums
# (https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)
# Dec 2018
# Example: Gradient flow in network
# writer.add_figure('training/gradient_flow',
#                   plot_grad_flow(agent.online.named_parameters(),
#                                  episode),
#                   global_step=episode)
def plot_grad_flow(named_parameters):
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


def append_timestamp(string, fmt_string=None):
    now = datetime.now()
    if fmt_string:
        return string + "_" + now.strftime(fmt_string)
    else:
        return string + "_" + str(now).replace(" ", "_")
