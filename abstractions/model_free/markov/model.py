import os

import torch
import numpy as np

from ...common.modules import MarkovHead, build_phi_network

class FeatureNet(torch.nn.Module):
    def __init__(self, args, num_actions, input_shape):
        super(FeatureNet, self).__init__()
        self.phi, self.feature_size = build_phi_network(args, input_shape)
        self.markov_head = MarkovHead(args, self.feature_size, num_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, x):
        return self.phi(x)

    def loss(self, batch):
        states, actions, _, next_states, _ = batch
        markov_loss = self.markov_head.compute_markov_loss(
            z0=self.phi(torch.as_tensor(states.astype(np.float32))),
            z1=self.phi(torch.as_tensor(next_states.astype(np.float32))),
            a=torch.as_tensor(actions, dtype=torch.int64),
        )
        loss = markov_loss
        return loss

    def save_phi(self, path, name):
        full_path = os.path.join(path, name)
        torch.save((self.phi, self.feature_size), full_path)
        return full_path

    def train_one_batch(self, batch):
        loss = self.loss(batch)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

