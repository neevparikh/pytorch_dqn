import gym 
import torch
import numpy as np

class LearnedEnv(gym.Env):

    """ Gym wrapper around learned model. """

    def __init__(self, action_space, state_space, model_path, start_space=None):
        super(LearnedEnv, self).__init__()
        self.action_space = action_space
        self.observation_space = state_space
        self.model = torch.load(model_path)
        self.model.eval()
        if start_space is None:
            self.start_space = state_space
        else:
            self.start_space = start_space
        self.state = None

    def step(self, action):
        assert self.state is not None, "Please call reset() before calling step()"
        with torch.no_grad():
            action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            state = torch.as_tensor(self.state, dtype=torch.float32).unsqueeze(0)
            next_state, reward, done = self.model(state, action)
        next_state = np.squeeze(next_state.numpy())
        reward = reward.item()
        done = int(round(done.item()))
        self.state = next_state
        return next_state, reward, done, {}
        

    def reset(self):
        state = self.start_space.sample()
        self.state = state
        return state
    
    def render(self):
        pass
