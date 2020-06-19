import argparse
from datetime import datetime
import random

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from atariari.benchmark.wrapper import AtariARIWrapper

from gym_wrappers import AtariPreprocess, MaxAndSkipEnv, FrameStack, ResetARI, \
        ObservationDictToInfo, ResizeObservation

def parse_args():
    # Parse input arguments
    # Use --help to see a pretty description of the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, required=True,
                        help='The gym environment to train on')
    parser.add_argument('--ari', action='store_true', required=False,
                        help='Whether to use annotated RAM')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'mlp'], required=True,
                        default='mlp', help="Type of architecture")
    parser.add_argument('--model-shape', type=str, choices=['small', 'medium', 'large', 'giant']),
                        default='medium', help="Shape of architecture (mlp only)")
    parser.add_argument('--gamma', type=float, required=False,
                        default=0.99, help='Gamma parameter')
    parser.add_argument('--model-path', type=str, required=False,
                        help='The path to the save the pytorch model')
    parser.add_argument('--output-path', type=str, required=False,
                        help='The output directory to store training stats')
    parser.add_argument('--load-checkpoint-path', type=str, required=False,
                        help='Path to checkpoint')
    parser.add_argument('--no-atari', action='store_true', required=False,
                        help='Do not use atari preprocessing')
    parser.add_argument('--no-tensorboard', action='store_true', required=False,
                        help='Do not use Tensorboard')
    parser.add_argument('--gpu', action='store_true', required=False,
                        help='Use the gpu or not')
    parser.add_argument('--render', action='store_true', required=False,
                        help='Render visual or not')
    parser.add_argument('--render-episodes', type=int, required=False,
                        default=5, help='Render every these many episodes')
    parser.add_argument('--num-frames', type=int, required=False,
                        default=4, help='Number of frames to stack (CNN only)')
    parser.add_argument('--max-steps', type=lambda x: int(float(x)), required=False,
                        default=40000, help='Number of steps to run for')
    parser.add_argument('--checkpoint-steps', type=lambda x: int(float(x)), required=False,
                        default=20000, help='Checkpoint every so often')
    parser.add_argument('--test-policy-steps', type=lambda x: int(float(x)), required=False,
                        default=1000, help='Policy is tested every these many steps')
    parser.add_argument('--warmup-period', type=lambda x: int(float(x)), required=False,
                        default=2000, help='Number of steps to act randomly and not train')
    parser.add_argument('--batchsize', type=int, required=False,
                        default=32, help='Number of experiences sampled from replay buffer')
    parser.add_argument('--gradient-clip', type=float, required=False,
                        default=0, help='How much to clip the gradients by, 0 is none')
    parser.add_argument('--reward-clip', type=float, required=False,
                        default=0, help='How much to clip reward, i.e. [-rc, rc]; 0 is unclipped')
    parser.add_argument('--epsilon-decay', type=lambda x: int(float(x)), required=False,
                        default=5000, help='Parameter for epsilon decay')
    parser.add_argument('--epsilon-decay-end', type=float, required=False,
                        default=0.05, help='Parameter for epsilon decay end')
    parser.add_argument('--replay-buffer-size', type=lambda x: int(float(x)), required=False,
                        default=50000, help='Max size of replay buffer')
    parser.add_argument('--lr', type=float, required=False,
                        default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--target-moving-average', type=float, required=False,
                        default=0.01, help='EMA parameter for target network')
    parser.add_argument('--vanilla-DQN', action='store_true', required=False,
                        help='Use the vanilla dqn update instead of double DQN')
    parser.add_argument('--seed', type=int, required=False,
                        default=10, help='The random seed for this run')
    parser.add_argument('--run-tag', type=str, required=True,
                        help="Run tag for the experient run.")

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

def make_atari(env, num_frames):
    """ Wrap env in atari processed env """
    return FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4), num_frames)


def make_ari(env):
    """ Wrap env in reset to match observation """
    return ResetARI(AtariARIWrapper(env))


def make_visual(env, shape):
    """ Wrap env to return pixel observations """
    env = PixelObservationWrapper(env, pixels_only=False, pixel_keys=("pixels",))
    env = ObservationDictToInfo(env, "pixels")
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    return env


def initialize_environment(args):
    # Initialize environment
    visual_cartpole_shape = (80, 120)
    if args.env == "VisualCartPole-v0":
        from pyvirtualdisplay import Display
        display = Display(visible=False, backend='xvfb').start()
        env = gym.make("CartPole-v0")
        test_env = gym.make("CartPole-v0")
        env.reset()
        test_env.reset()
        env = make_visual(env, visual_cartpole_shape)
        test_env = make_visual(env, visual_cartpole_shape)
    elif args.env == "VisualCartPole-v1":
        from pyvirtualdisplay import Display
        display = Display(visible=False, backend='xvfb').start()
        env = gym.make("CartPole-v1")
        test_env = gym.make("CartPole-v1")
        env.reset()
        test_env.reset()
        env = make_visual(env, visual_cartpole_shape)
        test_env = make_visual(env, visual_cartpole_shape)
    else:
        env = gym.make(args.env)
        test_env = gym.make(args.env)

    if args.model_type == 'cnn':
        assert args.num_frames
        if not args.no_atari:
            print("Using atari preprocessing")
            env = make_atari(env, args.num_frames)
            test_env = make_atari(test_env, args.num_frames)
        else:
            print("FrameStacking with {}".format(args.num_frames))
            env = FrameStack(env, args.num_frames)
            test_env = FrameStack(test_env, args.num_frames)

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

