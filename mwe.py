from collections import deque
# from pympler.asizeof import asizeof as aso
import numpy as np
import random
from utils import make_atari
from model import Experience
import gym
import torch

from atari_wrappers import LazyFrames, AtariPreprocess, MaxAndSkipEnv, FrameStack
from memory import SequentialMemory

size = 5000
n_steps = 10000
# rb = deque(maxlen=size)
rb = SequentialMemory(limit=size)

#env = make_atari(gym.make("PongNoFrameskip-v4"),4)
env = gym.make("PongNoFrameskip-v4")
env = AtariPreprocess(env)
env = MaxAndSkipEnv(env, 4)
# env = FrameStack(env, 4)
# env = FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4), 4)

def test_func():
    state = env.reset()
    for i in range(n_steps):
        print(i)
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        rb.append(Experience(state, action, reward, next_state, int(done)))
        state = next_state

        if i > 32:
            minibatch = rb.sample(32)
            minibatch = Experience(*zip(*minibatch))
            a = torch.FloatTensor(np.array(minibatch.state))
            b = torch.FloatTensor(np.array(minibatch.next_state))

test_func()
