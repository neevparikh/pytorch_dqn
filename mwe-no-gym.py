from collections import deque
# from pympler.asizeof import asizeof as aso
import numpy as np
import random
import torch

from atari_wrappers import LazyFrames

size = 5000
n_steps = 10000
rb = deque(maxlen=size)

two_buf = np.zeros(2,84,84)

def test_func_nogym():
    for i in range(n_steps):
        print(i)
        for j in range(4):
            obs = np.random.rand(84, 84, dtype=np.uint8)
            if j == 2:
                two_buf[0] = obs
            if j == 3:
                two_buf[1] = obs
            max_obs = two_buf.max(axis=0)
        rb.append(LazyFrames([max_obs]))
        if i > 32:
            minibatch = random.sample(rb, 32)
            a = torch.FloatTensor(np.array(minibatch))
            b = torch.FloatTensor(np.array(minibatch))

test_func_nogym()
