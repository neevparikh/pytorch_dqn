import gym

from ..model_based.env import LearnedEnv

cartpole = gym.make("CartPole-v1")

env = LearnedEnv(cartpole.action_space, cartpole.observation_space,
        'cartpole_learned/test_model_based_2020-06-28_10:25:04.011339.pth',
        gym.spaces.Box(shape=(4,), high=0.05, low=-0.05))

state = env.reset()
done = False
steps = 0

while not done and steps < 100:
    state, reward, done, _ = env.step(env.action_space.sample())
    print(state, reward, done)

