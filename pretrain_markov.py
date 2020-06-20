import argparse
import json
import os
import sys
import random
import pickle
from collections import namedtuple

import gym
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils import initialize_environment, reset_seeds, parse_args
from model import FeatureNet
from gym_wrappers import FrameStack, MaxAndSkipEnv, AtariPreprocess
from replay_buffer import Experience, ReplayBuffer

parser = parse_args(no_parse=True)
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE',
                    help='Network hidden size')
# yapf: disable
# parser.add_argument('--env', type=str, required=True, help='ATARI game')
# parser.add_argument('--gpu', action='store_true', help='Disable CUDA')
# parser.add_argument('--max-steps', type=int, default=int(15e3), metavar='STEPS',
#                       help='Number of gradient steps')
# parser.add_argument('--num-frames', type=int, default=4, metavar='T',
#                     help='Number of consecutive states processed')
# parser.add_argument('--replay-buffer-size', type=int, default=int(20e3), metavar='CAPACITY',
#                     help='Experience memory capacity')
# parser.add_argument('--lr', type=float, default=0.003, metavar='η',
#                     help='Learning rate')
# parser.add_argument('--batchsize', type=int, default=2048, metavar='SIZE', help='Batch size')
# parser.add_argument('--seed', type=int, default=10, help='Random seed')
# parser.add_argument('--run-tag', default='env', type=str, required=True,
#                     help='Run Tag for the experient run.')
# yapf: enable

# Setup
args = parser.parse_args()
args.overfit_one_batch = False

# Check if GPU can be used and was asked for
if args.gpu and torch.cuda.is_available():
    args.device = torch.device('cuda:0')
else:
    args.device = torch.device('cpu')

# Set seeds
reset_seeds(args.seed)



def generate_experiences(args, env):
    mem = ReplayBuffer(args.replay_buffer_size)

    state, done = env.reset(), False
    i = 0
    pbar = tqdm(total=args.replay_buffer_size)
    while i < args.replay_buffer_size:
        if done:
            state, done = env.reset(), False
        action = np.random.randint(0, env.action_space.n)
        next_state, reward, done, _ = env.step(action)
        if not done:
            mem.append(state, action, reward, next_state, int(done))
            i += 1
            pbar.update(1)
        state = next_state
    return mem


def sample(mem, args):
    return mem.sample(args.batchsize)


def train():
    run_tag = args.run_tag
    results_dir = os.path.join('logs', 'pretraining', run_tag)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir,'params.json'), 'w') as fp:
        param_dict = vars(args)
        param_dict['device'] = str(args.device)
        json.dump(param_dict, fp)

    print('Making env')
    env = initialize_environment(args)[0]
    if len(env.observation_space.shape) > 2:
        input_shape = env.observation_space.shape[1:]
    else:
        input_shape = env.observation_space.shape
    network = FeatureNet(args, env.action_space.n, input_shape)

    experiences_filepath = os.path.join(results_dir, 'experiences.mem')
    print('Generating experiences')
    mem = generate_experiences(args, env)

    print('Training')
    with open(os.path.join(results_dir,'loss.csv'), 'w') as fp:
        fp.write('step,loss\n')  # write headers

        batch = sample(mem, args)
        for step in tqdm(range(args.max_steps)):
            if not args.overfit_one_batch and step > 0:
                batch = sample(mem, args)
            loss = network.train_one_batch(batch)
            fp.write(f"{step},{loss}\n")
            fp.flush()

    phi_net_path = network.save_phi(results_dir, 'phi_model.pth')
    print('Saved phi network to {}'.format(phi_net_path))


if __name__ == '__main__':
    train()
