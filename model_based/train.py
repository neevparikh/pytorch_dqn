import time

import gym

from common.replay_buffer import ReplayBuffer

env = gym.make("CartPole-v1")

# Episode loop
global_steps = 0
steps = 1
episode = 0
start = time.time()
t_zero = time.time()
             
end = time.time() + 1

score = 0
while global_steps < args.max_steps:
    info_str = "episode: {}, ".format(episode)
    info_str += "steps: {}, ".format(global_steps)
    info_str += "ep_score: {}, ".format(score)
    info_str += "FPS: {}".format(steps/(end - start))
    print(info_str)
    start = time.time()

    state = env.reset()
    done = False
