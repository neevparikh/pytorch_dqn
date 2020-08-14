from datetime import datetime
import random
import math

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from atariari.benchmark.wrapper import AtariARIWrapper

from .gym_wrappers import AtariPreprocess, MaxAndSkipEnv, FrameStack, ResetARI, \
        ObservationDictToInfo, ResizeObservation, IndexedObservation

## DQN utils ##


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        torch.nn.init.uniform_(m.bias, -1, 1)


def conv2d_size_out(size, kernel_size, stride):
    ''' Adapted from pytorch tutorials: 
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
    '''
    return ((size[0] - (kernel_size[0] - 1) - 1) // stride + 1,
            (size[1] - (kernel_size[1] - 1) - 1) // stride + 1)


def deque_to_tensor(last_num_frames):
    """ Convert deque of n frames to tensor """
    return torch.cat(list(last_num_frames), dim=0)


def plot_grad_flow(named_parameters):
    '''
    Thanks to RoshanRane - Pytorch forums (Dec 2018)
        - (https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    writer.add_figure('training/gradient_flow', plot_grad_flow(agent.online.named_parameters(), 
        episode), global_step=episode)

    '''
    plt.clf()
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
    plt.hlines(0, 0, len(ave_grads) + 1, lw=4, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0, top=2.5)  # zoom in on the lower gradient regions
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


def make_atari(env, num_frames, action_stack=False):
    """ Wrap env in atari processed env """
    try:
        noop_action = env.get_action_meanings().index("NOOP")
    except ValueError:
        print("Cannot find NOOP in env, defaulting to 0")
        noop_action = 0
    return FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4),
                      num_frames,
                      action_stack=action_stack,
                      reset_action=noop_action)


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


def get_wrapped_env(env_string, wrapper_func, fake_display=True, **kwargs):
    if fake_display:
        from pyvirtualdisplay import Display
        _ = Display(visible=False, backend='xvfb').start()
    env = gym.make(env_string)
    test_env = gym.make(env_string)
    env.reset()
    test_env.reset()
    env = wrapper_func(env, **kwargs)
    test_env = wrapper_func(test_env, **kwargs)
    return env, test_env


def initialize_environment(args):
    # Initialize environment
    visual_cartpole_shape = (80, 120)
    visual_pendulum_shape = (120, 120)
    if args.env == "VisualCartPole-v0":
        env, test_env = get_wrapped_env("CartPole-v0", make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualCartPole-v1":
        env, test_env = get_wrapped_env("CartPole-v1", make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualPendulum-v0":
        env, test_env = get_wrapped_env("Pendulum-v0", make_visual, shape=visual_pendulum_shape)
    elif args.env == "CartPole-PosAngle-v0":
        env, test_env = get_wrapped_env("CartPole-v0", IndexedObservation, indices=[0, 1],
                fake_display=False)
    elif args.env == "CartPole-PosAngle-v1":
        env, test_env = get_wrapped_env("CartPole-v1", IndexedObservation, indices=[0, 1],
                fake_display=False)
    elif args.env == "VisualMountainCar-v0":
        env, test_env = get_wrapped_env("MountainCar-v0", make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualAcrobot-v1":
        env, test_env = get_wrapped_env("Acrobot-v1", make_visual, shape=visual_pendulum_shape)
    elif args.env[:6] == 'Visual':
        env, test_env = get_wrapped_env(args.env[6:], make_visual, shape=(64,64))
    else:
        env = gym.make(args.env)
        test_env = gym.make(args.env)

    if args.model_type == 'cnn':
        assert args.num_frames

        if args.action_stack:
            print("Stacking actions")
            args.num_frames *= 2

        if not args.no_atari:
            print("Using atari preprocessing")
            env = make_atari(env, args.num_frames, action_stack=args.action_stack)
            test_env = make_atari(test_env, args.num_frames, action_stack=args.action_stack)
        else:
            print("FrameStacking with {}".format(args.num_frames))
            env = FrameStack(env, args.num_frames, action_stack=args.action_stack, reset_action=0)
            test_env = FrameStack(test_env,
                                  args.num_frames,
                                  action_stack=args.action_stack,
                                  reset_action=0)

    if args.ari:
        print("Using ARI")
        env = make_ari(env)
        test_env = make_ari(test_env)

    env.seed(args.seed)
    test_env.seed(args.seed + 1000)
    return env, test_env


## SAC utils ##


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
