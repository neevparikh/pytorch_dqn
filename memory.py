# Adapted from https://github.com/keras-rl/keras-rl/blob/d9e3b64a20f056030c02bfe217085b7e54098e48/rl/memory.py
#
# Original code license is retained:
#
# The MIT License (MIT)
#
# Copyright (c) 2016 Matthias Plappert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import namedtuple
import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, terminal, state1')

class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class SequentialMemory(object):
    def __init__(self, limit, window_length=4):
        self.limit = limit
        self.window_length = window_length

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, window_length=None):
        if window_length is None:
            window_length=self.window_length
        # Draw random indexes such that we have at least `window_length` entries before each index.
        batch_idxs = np.random.random_integers(window_length, self.nb_entries - 1, size=batch_size)
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for current_idx in range(idx - 2, idx - window_length - 1, -1):
                if self.terminals[current_idx]:
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < window_length:
                state0.insert(0, np.copy(state0[0]))

            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            if terminal:
                state1.append(np.copy(state1[-1]))
            else:
                state1.append(self.observations[idx])

            assert len(state0) == window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0, action, reward, terminal, state1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal):
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and whether the next state is `terminal` or not.
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)
