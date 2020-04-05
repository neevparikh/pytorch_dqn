import torch
import numpy as np
import random
from collections import deque, namedtuple

from utils import sync_networks

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class DQN_model(torch.nn.Module):
    """Docstring for DQN model """

    def __init__(self, device, state_space, action_space, num_actions):
        """Defining DQN model
        """
        # initialize all parameters
        super(DQN_model, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.device = device
        self.num_actions = num_actions

        # architecture
        self.layer_sizes = [(512, 512), (512, 256), (256, 128)]

        self.build_model()

    def build_model(self):
        # output should be in batchsize x num_actions
        # First layer takes in states
        layers = [
            torch.nn.Linear(self.state_space.shape[0], self.layer_sizes[0][0]),
            torch.nn.ReLU()
        ]
        for size in self.layer_sizes:
            layer = [torch.nn.Linear(size[0], size[1]), torch.nn.ReLU()]
            layers.extend(layer)

        layers.append(torch.nn.Linear(self.layer_sizes[-1][1],
                                      self.num_actions))

        self.body = torch.nn.Sequential(*layers)

    def forward(self, state):
        q_value = self.body(state)
        return q_value

    def max_over_actions(self, state):
        state = state.to(self.device)
        return torch.max(self(state), dim=1)

    def argmax_over_actions(self, state):
        state = state.to(self.device)
        return torch.argmax(self(state), dim=1)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.Tensor(state).unsqueeze(0)
                action_tensor = self.argmax_over_actions(state_tensor)
                action = action_tensor.cpu().detach().numpy().flatten()[0]
                assert self.action_space.contains(action)
            return action


class DQN_agent:
    """Docstring for DQN agent """

    def __init__(self, device, state_space, action_space, num_actions,
                 target_moving_average, gamma, replay_buffer_size,
                 epsilon_decay, epsilon_decay_start, double_DQN):
        """Defining DQN model
        """
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.online = DQN_model(device, state_space, action_space, num_actions)
        self.online = self.online.to(device)
        self.target = DQN_model(device, state_space, action_space, num_actions)
        self.target = self.target.to(device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.gamma = gamma
        self.target_moving_average = target_moving_average
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_start = epsilon_decay_start
        self.device = device
        
        self.double_DQN = double_DQN

    def loss_func(self, minibatch, writer=None, writer_step=None):
        # Make tensors
        state_tensor = torch.Tensor(minibatch.state).to(self.device)
        next_state_tensor = torch.Tensor(minibatch.next_state).to(self.device)
        action_tensor = torch.Tensor(minibatch.action).to(self.device)
        reward_tensor = torch.Tensor(minibatch.reward).to(self.device)
        done_tensor = torch.Tensor(minibatch.done).to(self.device)

        # Get q value predictions
        q_pred_batch = self.online(state_tensor).gather(
            dim=1, index=action_tensor.long().unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_DQN:
                selected_actions = self.online.argmax_over_actions(
                    next_state_tensor)
                q_target = self.target(next_state_tensor).gather(
                    dim=1,
                    index=selected_actions.long().unsqueeze(1)).squeeze(1)
            else:
                q_target = self.target.max_over_actions(
                    next_state_tensor.detach()).values

        q_label_batch = reward_tensor + (self.gamma) * (1 -
                                                        done_tensor) * q_target
        q_label_batch = q_label_batch.detach()

        # Logging
        if writer:
            writer.add_scalar('training/avg_q_label', q_label_batch.mean(),
                              writer_step)
            writer.add_scalar('training/avg_q_pred', q_pred_batch.mean(),
                              writer_step)
            writer.add_scalar('training/avg_reward', reward_tensor.mean(),
                              writer_step)
        return torch.nn.functional.mse_loss(q_label_batch, q_pred_batch)

    def sync_networks(self):
        sync_networks(self.target, self.online, self.target_moving_average)

    def set_epsilon(self, episode, writer=None):
        if episode > self.epsilon_decay_start:
            self.online.epsilon = (1 + episode - self.epsilon_decay_start)**(
                -1 / self.epsilon_decay)
            self.target.epsilon = (1 + episode - self.epsilon_decay_start)**(
                -1 / self.epsilon_decay)
        else:
            self.online.epsilon = 1
            self.target.epsilon = 1
        if writer:
            writer.add_scalar('training/epsilon', self.online.epsilon, episode)
