import torch

from ...common.modules import ActorCritic
from ...common.utils import hard_update


class Rollouts:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
    
    def __len__(self):
        return len(self.actions)

    def clear_rollouts(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


class PPO:
    def __init__(self,
                 state_space,
                 action_space,
                 device,
                 discrete,
                 lr,
                 gamma,
                 gradient_updates,
                 eps_clip,
                 **kwargs):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gradient_updates = gradient_updates
        self.discrete = discrete
        self.device = device

        if discrete:
            action_dim = action_space.n
            state_dim = state_space.shape[0] # TODO - modify for CNN version
        else:
            action_dim = action_space.shape[0]
            state_dim = state_space.shape[0] # TODO - modify for CNN version


        self.rollouts = Rollouts()
        self.test_rollouts = Rollouts()

        self.policy = ActorCritic(state_dim, action_dim, discrete, device, **kwargs).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, action_dim, discrete, device, **kwargs).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def act(self, state, test=False):
        rollout = self.test_rollouts if test else self.rollouts
        if self.discrete:
            return self.policy_old.act(state, rollout)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            return self.policy_old.act(state, rollout).cpu().data.numpy().flatten()

    def update(self):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rollouts.rewards), reversed(self.rollouts.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        if self.discrete:
            old_states = torch.stack(self.rollouts.states).to(self.device).detach()
            old_actions = torch.stack(self.rollouts.actions).to(self.device).detach()
            old_logprobs = torch.stack(self.rollouts.logprobs).to(self.device).detach()
        else:
            old_states = torch.squeeze(torch.stack(self.rollouts.states).to(self.device),
                                       1).detach()
            old_actions = torch.squeeze(torch.stack(self.rollouts.actions).to(self.device),
                                        1).detach()
            old_logprobs = torch.squeeze(torch.stack(self.rollouts.logprobs),
                                         1).to(self.device).detach()

        cumulative_loss = 0

        # Optimize policy for gradient_updates:
        for _ in range(self.gradient_updates):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * torch.nn.functional.mse_loss(
                state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            cumulative_loss += loss.mean().item()

        # Copy new weights into old policy:
        # self.policy_old.load_state_dict(self.policy.state_dict())
        hard_update(self.policy_old, self.policy)
        self.rollouts.clear_rollouts()

        return cumulative_loss

