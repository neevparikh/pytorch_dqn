from collections import deque
# from pympler.asizeof import asizeof as aso
from atari_wrappers import LazyFrames
import numpy as np
import random
from utils import make_atari
from model import Experience
import gym
import torch

size = 500000
rb = deque(maxlen=size)
env = make_atari(gym.make("PongNoFrameskip-v4"),4)

def test_func():
    state = env.reset()
    for i in range(size):
        print(i)
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        rb.append(Experience(state, action, reward, next_state, int(done)))
        state = next_state

        if i > 20:
            minibatch = random.sample(rb, 5)
            minibatch = Experience(*zip(*minibatch))
            a = torch.FloatTensor(np.array(minibatch.state))
            b = torch.FloatTensor(np.array(minibatch.next_state))

test_func()
