import torch

from ..common.utils import plot_grad_flow
from ..common.modules import MLP_Body, CNN_Body

class ModelNet(torch.nn.Module):
    def __init__(self, args, state_space, num_actions):
        super(ModelNet, self).__init__()
        self.body = MLP_Body(state_space, num_actions, args.model_shape)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, state, action, reward):
        pass

    def loss(self, batch, writer=None, writer_step=None):
        pred_state = self(batch.state, batch.action, batch.reward)
        loss = torch.nn.functional.mse_loss(pred_state, batch.next_state)

        if writer:
            writer.add_scalar('training/step_loss', loss.mean(), writer_step)

        return loss

    def train_batch(self, batch, global_steps, writer, gradient_clip):
        loss = self.loss(batch, writer, global_steps)
        self.zero_grad()
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), gradient_clip)

        if writer and global_steps % 1000 == 0:
            writer.add_figure('gradient_flow', plot_grad_flow(self.online.named_parameters()),
                    global_steps)

        self.optimizer.step()

        return loss.detach().cpu().numpy()

