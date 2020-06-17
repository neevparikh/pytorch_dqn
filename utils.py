import argparse
from datetime import datetime
import random

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import gym

from gym_wrappers import AtariPreprocess, MaxAndSkipEnv, FrameStack, ResetARI
from atariari.benchmark.wrapper import AtariARIWrapper


def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env',
                        help='The gym environment to train on',
                        type=str,
                        required=True)
    parser.add_argument('--ari',
                        help='Whether to use annotated RAM',
                        action='store_true',
                        required=False)
    parser.add_argument('--model-type',
                        help="Type of architecture",
                        type=str,
                        default='mlp',
                        choices=['cnn', 'mlp'],
                        required=True)
    parser.add_argument('--model-shape',
                        help="Shape of architecture (mlp only)",
                        type=str,
                        default='medium',
                        choices=['small', 'medium', 'large'],
                        required=True)
    parser.add_argument('--gamma',
                        help='Gamma parameter',
                        type=float,
                        default=0.99,
                        required=False)
    parser.add_argument('--model-path',
                        help='The path to the save the pytorch model',
                        type=str,
                        required=False)
    parser.add_argument('--output-path',
                        help='The output directory to store training stats',
                        type=str,
                        required=False)
    parser.add_argument('--load-checkpoint-path',
                        help='Path to checkpoint',
                        type=str,
                        required=False)
    parser.add_argument('--no-atari',
                        help='Do not use atari preprocessing',
                        action='store_true',
                        required=False)
    parser.add_argument('--no-tensorboard',
                        help='Do not use Tensorboard',
                        action='store_true',
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
                        type=lambda x: int(float(x)),
                        default=40000,
                        required=False)
    parser.add_argument('--checkpoint-steps',
                        help='Checkpoint every so often',
                        type=lambda x: int(float(x)),
                        default=20000,
                        required=False)
    parser.add_argument('--test-policy-steps',
                        help='Policy is tested every these many steps',
                        type=lambda x: int(float(x)),
                        default=1000,
                        required=False)
    parser.add_argument('--warmup-period',
                        help='Number of steps to act randomly and not train',
                        type=lambda x: int(float(x)),
                        default=2000,
                        required=False)
    parser.add_argument('--batchsize',
                        help='Number of experiences sampled from replay buffer',
                        type=int,
                        default=32,
                        required=False)
    parser.add_argument('--gradient-clip',
                        help='How much to clip the gradients by, 0 is none',
                        type=float,
                        default=0,
                        required=False)
    parser.add_argument('--reward-clip',
                        help='How much to clip reward, clipped in [-rc, rc], 0 is unclipped',
                        type=float,
                        default=0,
                        required=False)
    parser.add_argument('--epsilon-decay',
                        help='Parameter for epsilon decay',
                        type=lambda x: int(float(x)),
                        default=5000,
                        required=False)
    parser.add_argument('--epsilon-decay-end',
                        help='Parameter for epsilon decay end',
                        type=float,
                        default=0.05,
                        required=False)
    parser.add_argument('--replay-buffer-size',
                        help='Max size of replay buffer',
                        type=lambda x: int(float(x)),
                        default=50000,
                        required=False)
    parser.add_argument('--lr',
                        help='Learning rate for the optimizer',
                        type=float,
                        default=0.001,
                        required=False)
    parser.add_argument('--target-moving-average',
                        help='EMA parameter for target network',
                        type=float,
                        default=0.01,
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
    parser.add_argument('--run-tag',
                        help="Run tag for the experient run.",
                        type=str,
                        required=True)

    return parser.parse_args()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        torch.nn.init.uniform_(m.bias, -1, 1)


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
    return FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4), num_frames)


def make_ari(env):
    """ Wrap env in reset to match observation """
    return ResetARI(AtariARIWrapper(env))


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

def reset_seeds(seed):
    # Setting cuda seeds
    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

    # Setting random seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def append_timestamp(string, fmt_string=None):
    now = datetime.now()
    if fmt_string:
        return string + "_" + now.strftime(fmt_string)
    else:
        return string + "_" + str(now).replace(" ", "_")

def initialize_environment(args):
    # Initialize environment
    env = gym.make(args.env)
    test_env = gym.make(args.env)
    if args.model_type == 'cnn':
        assert args.num_frames
        if not args.no_atari:
            print("Using atari preprocessing")
            env = make_atari(env, args.num_frames)
            test_env = make_atari(test_env, args.num_frames)

    if args.ari:
        print("Using ARI")
        env = make_ari(env)
        test_env = make_ari(test_env)

    if type(env.action_space) != gym.spaces.Discrete:
        raise NotImplementedError("DQN for continuous action_spaces hasn't been\
                implemented")
    env.seed(args.seed)
    test_env.seed(args.seed + 1000)
    return env, test_env

