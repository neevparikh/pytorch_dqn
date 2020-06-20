from collections import deque

import numpy as np
import torchvision.transforms as T
import gym
import cv2

class IndexedObservation(gym.ObservationWrapper):
    r""" Return elements of observation at given indices """
    def __init__(self, env, indices):
        super(IndexedObservation, self).__init__(env)
        self.indices = indices

        assert len(env.observation_space.shape) == 1, env.observation_space
        wrapped_obs_len = env.observation_space.shape[0]
        assert len(indices) <= wrapped_obs_len, indices
        assert all(i < wrapped_obs_len for i in indices), indices
        self.observation_space = gym.spaces.Box(low=env.observation_space.low[indices],
                                                high=env.observation_space.high[indices],
                                                dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation[self.indices]


# Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/resize_observation.py
class ResizeObservation(gym.ObservationWrapper):
    r"""Downsample the image observation to a square image. """
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        return observation


class ObservationDictToInfo(gym.Wrapper):
    def __init__(self, env, state_key):
        gym.Wrapper.__init__(self, env)
        assert type(env.observation_space) == gym.spaces.Dict
        self.observation_space = env.observation_space.spaces[state_key]
        self.state_key = state_key

    def reset(self, **kwargs):
        next_state_as_dict = self.env.reset(**kwargs)
        return next_state_as_dict[self.state_key]

    def step(self, action):
        next_state_as_dict, reward, done, info = self.env.step(action)
        info.update(next_state_as_dict)
        return next_state_as_dict[self.state_key], reward, done, info


class ResetARI(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # change the observation space to accurately represent
        # the shape of the labeled RAM observations
        self.observation_space = gym.spaces.Box(
            0,
            255,  # max value
            shape=(len(self.env.labels()),),
            dtype=np.uint8)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # reset the env and get the current labeled RAM
        return np.array(list(self.env.labels().values()))

    def step(self, action):
        # we don't need the obs here, just the labels in info
        _, reward, done, info = self.env.step(action)
        # grab the labeled RAM out of info and put as next_state
        next_state = np.array(list(info['labels'].values()))
        return next_state, reward, done, info


# Adapted from OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class AtariPreprocess(gym.Wrapper):
    def __init__(self, env, shape=(84, 84)):
        """ Preprocessing as described in the Nature DQN paper (Mnih 2015) """
        gym.Wrapper.__init__(self, env)
        self.shape = shape
        self.transforms = T.Compose([
            T.ToPILImage(mode='YCbCr'),
            T.Lambda(lambda img: img.split()[0]),
            T.Resize(self.shape),
            T.Lambda(lambda img: np.array(img, copy=False)),
        ])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        return self.transforms(self.env.reset(**kwargs))

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self.transforms(next_state), reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=((k,) + shp),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once.  It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers.  This object should only be
        converted to numpy array before being passed to the model."""
        self._frames = frames

    def _force(self):
        return np.stack(self._frames, axis=0)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]
