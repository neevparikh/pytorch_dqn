import gym
from torch.utils.tensorboard import SummaryWriter

env = gym.make('CartPole-v0')
writer = SummaryWriter()

steps = 0
episode = 0
while steps < 1000:
    done = False
    state = env.reset()
    cumulative_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        cumulative_reward += reward
        steps += 1
        episode += 1
    writer.add_scalar('random_agent/reward', cumulative_reward, steps)
