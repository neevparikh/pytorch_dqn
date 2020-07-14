import argparse
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

float_to_int = lambda x: int(float(x))

# yapf: disable
common_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
common_parser.add_argument('--env', type=str, required=True,
        help='The gym environment to train on')
common_parser.add_argument('--gamma', type=float, required=False, default=0.99,
        help='Gamma parameter')
common_parser.add_argument('--no-tensorboard', action='store_true', required=False,
        help='Do not use Tensorboard')
common_parser.add_argument('--gpu', action='store_true', required=False,
        help='Use the gpu or not')
common_parser.add_argument('--max-steps', type=float_to_int, required=False, default=40000,
        help='Number of steps to run for')
common_parser.add_argument('--model-path', type=str, required=False,
        help='The path to the save the pytorch model')
common_parser.add_argument('--output-path', default='logs', type=str, required=False,
        help='The output directory to store training stats')
common_parser.add_argument('--seed', type=int, required=False, default=10,
        help='The random seed for this run')
common_parser.add_argument('--run-tag', type=str, required=True,
        help="Run tag for the experient run.")
common_parser.add_argument('--render', action='store_true', required=False,
        help='Render visual or not')
common_parser.add_argument('--render-episodes', type=int, required=False, default=5,
        help='Render every these many episodes')
common_parser.add_argument('--replay-buffer-size', type=float_to_int, required=False, default=50000,
        help='Max size of replay buffer')
common_parser.add_argument('--lr', type=float, required=False, default=0.001,
        help='Learning rate for the optimizer')
common_parser.add_argument('--batchsize', type=int, required=False, default=32,
        help='Number of experiences sampled from replay buffer')
common_parser.add_argument('--gradient-clip', type=float, required=False, default=0,
        help='How much to clip the gradients by, 0 is none')
common_parser.add_argument('--reward-clip', type=float, required=False, default=0,
        help='How much to clip reward, i.e. [-rc, rc]; 0 is unclipped')
common_parser.add_argument('--model-shape', type=str, default='medium',
        choices=['tiny', 'small', 'medium', 'large', 'giant'],
        help="Shape of architecture (mlp only)")
common_parser.add_argument('--num-frames', type=int, required=False, default=4,
        help='Number of frames to stack (CNN only)')
common_parser.add_argument('--warmup-period', type=float_to_int, required=False, default=2000,
        help='Number of steps to act randomly and not train')
common_parser.add_argument('--episodes-per-eval', type=int, default=10,
        help='Number of episodes per evaluation (i.e. during test)')

dqn_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
dqn_parser.add_argument('--ari', action='store_true', required=False,
        help='Whether to use annotated RAM')
dqn_parser.add_argument('--model-type', type=str, required=True, default='mlp',
        choices=['cnn', 'mlp'],
        help="Type of architecture")
dqn_parser.add_argument('--load-checkpoint-path', type=str, required=False,
        help='Path to checkpoint')
dqn_parser.add_argument('--no-atari', action='store_true', required=False,
        help='Do not use atari preprocessing')
dqn_parser.add_argument('--checkpoint-steps', type=float_to_int, required=False, default=20000,
        help='Checkpoint every so often')
dqn_parser.add_argument('--test-policy-steps', type=float_to_int, required=False, default=1000,
        help='Policy is tested every these many steps')
dqn_parser.add_argument('--epsilon-decay-length', type=float_to_int, required=False, default=5000,
        help='Number of steps to linearly decay epsilon')
dqn_parser.add_argument('--final-epsilon-value', type=float, required=False, default=0.05,
        help='Final epsilon value, between 0 and 1')
dqn_parser.add_argument('--target-moving-average', type=float, required=False, default=0.01,
        help='EMA parameter for target network')
dqn_parser.add_argument('--vanilla-DQN', action='store_true', required=False,
        help='Use the vanilla dqn update instead of double DQN')

sac_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
sac_parser.add_argument('--policy', default="Gaussian", choices=["Gaussian", "Deterministic"],
        help='Policy Type')
sac_parser.add_argument('--target-moving-average', type=float, required=False, default=0.005,
        help='EMA parameter for target network')
sac_parser.add_argument('--alpha', type=float, default=0.2,
        help='Temperature parameter α determines the relative importance of the entropy term \
                against the reward (default: 0.2)')
sac_parser.add_argument('--automatic-entropy-tuning', action='store_true',
        help='Automaically adjust α (default: False)')
sac_parser.add_argument('--hidden-size', type=int, default=256,
        help='hidden size (default: 256)')
sac_parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'cnn'],
        help="Type of architecture")
sac_parser.add_argument('--updates-per-step', type=int, default=1,
        help='model updates per simulator step (default: 1)')
sac_parser.add_argument('--target-update-interval', type=int, default=1,
        help='Value target update per no. of updates per step (default: 1)')
sac_parser.add_argument('--test-policy-steps', type=float_to_int, required=False, default=1000,
        help='Policy is tested every these many steps')
sac_parser.add_argument('--no-atari', action='store_true', required=False,
        help='Do not use atari preprocessing')
sac_parser.add_argument('--ari', action='store_true', required=False,
        help='Whether to use annotated RAM')

ppo_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
ppo_parser.add_argument('--discrete', action='store_true', required=False,
        help='Discrete action space')
ppo_parser.add_argument('--gradient-updates', type=float_to_int, default=80,
        help='model updates per simulator step')
ppo_parser.add_argument('--update-frequency', type=float_to_int, default=4e3,
        help='number of steps between model updates')
ppo_parser.add_argument('--action-std', type=float, default=0.5,
        help='standard deviation for action (continuous only)')
ppo_parser.add_argument('--eps-clip', type=float, default=0.2,
        help='Eps clip parameter for PPO')
ppo_parser.add_argument('--hidden-size', type=int, default=256,
        help='hidden size (default: 256)')
ppo_parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp'],
        help="Type of architecture")
ppo_parser.add_argument('--no-atari', action='store_true', required=False,
        help='Do not use atari preprocessing')
ppo_parser.add_argument('--ari', action='store_true', required=False,
        help='Whether to use annotated RAM')
ppo_parser.add_argument('--test-policy-steps', type=float_to_int, required=False, default=1000,
        help='Policy is tested every these many steps')

model_based_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
# yapf: enable


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
    visual_pendulum_shape = (120, 120)
    if args.env == "VisualCartPole-v0":
        from pyvirtualdisplay import Display
        _ = Display(visible=False, backend='xvfb').start()
        env = gym.make("CartPole-v0")
        test_env = gym.make("CartPole-v0")
        env.reset()
        test_env.reset()
        env = make_visual(env, visual_cartpole_shape)
        test_env = make_visual(env, visual_cartpole_shape)
    elif args.env == "VisualCartPole-v1":
        from pyvirtualdisplay import Display
        _ = Display(visible=False, backend='xvfb').start()
        env = gym.make("CartPole-v1")
        test_env = gym.make("CartPole-v1")
        env.reset()
        test_env.reset()
        env = make_visual(env, visual_cartpole_shape)
        test_env = make_visual(env, visual_cartpole_shape)
    elif args.env == "VisualPendulum-v0":
        from pyvirtualdisplay import Display
        _ = Display(visible=False, backend='xvfb').start()
        env = gym.make("Pendulum-v0")
        test_env = gym.make("Pendulum-v0")
        env.reset()
        test_env.reset()
        env = make_visual(env, visual_pendulum_shape)
        test_env = make_visual(env, visual_pendulum_shape)
    elif args.env == "CartPole-PosAngle-v0":
        env = IndexedObservation(gym.make("CartPole-v0"), [0, 1])
        test_env = IndexedObservation(gym.make("CartPole-v0"), [0, 1])
    elif args.env == "CartPole-PosAngle-v1":
        env = IndexedObservation(gym.make("CartPole-v1"), [0, 1])
        test_env = IndexedObservation(gym.make("CartPole-v1"), [0, 1])
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
