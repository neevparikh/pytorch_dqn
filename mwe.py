from collections import deque
# from pympler.asizeof import asizeof as aso
import numpy as np
import random
from utils import make_atari
from model import Experience
import gym
import torch

from atari_wrappers import LazyFrames, AtariPreprocess, MaxAndSkipEnv, FrameStack
from memory import ReplayBuffer

size = 5000
n_steps = 10000
# rb = deque(maxlen=size)
rb = ReplayBuffer(size)

env = make_atari(gym.make("PongNoFrameskip-v4"),4)
# env = gym.make("PongNoFrameskip-v4")
# env = AtariPreprocess(env)
# env = MaxAndSkipEnv(env, 4)
# env = FrameStack(env, 4)
# env = FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4), 4)

def test_func():
    state = env.reset()
    for i in range(n_steps):
        print(i)
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # rb.append(Experience(state, action, reward, next_state, done))
        rb.append(state, action, reward, next_state, done)
        state = next_state

        if i > 32:
            # minibatch = Experience(*zip(*random.sample(rb,32)))
            minibatch = Experience(*rb.sample(32))
            a = torch.as_tensor(minibatch.state,dtype=torch.float32)
            b = torch.as_tensor(minibatch.next_state,dtype=torch.float32)

test_func()
