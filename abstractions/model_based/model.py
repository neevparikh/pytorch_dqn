import torch

from ..common.utils import plot_grad_flow

class ModelNet(torch.nn.Module):
    def __init__(self, args, device, state_space, num_actions):
        super(ModelNet, self).__init__()
        self.state_space = state_space
        self.num_actions = num_actions
        self.device = device

        if args.model_shape == 'tiny':
            self.layer_sizes = [(15,),(15,)]
        elif args.model_shape == 'small':
            self.layer_sizes = [(16,), (16, 16), (16, 16), (16, 16), (16,)]
        elif args.model_shape == 'medium':
            self.layer_sizes = [(256,), (256, 256), (256, 256), (256, 256), (256,)]
        elif args.model_shape == 'large':
            self.layer_sizes = [(1024,), (1024, 1024), (1024, 1024), (1024, 1024), (1024, 1024),
                    (1024,)]
        elif args.model_shape == 'giant':
            self.layer_sizes = [(2048,), (2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048),
                                (2048, 2048), (2048, 2048), (2048,)]

        self.build_model()

        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)


    def build_model(self):
        layers = [
            torch.nn.Linear(self.state_space.shape[0] + 2, self.layer_sizes[0][0]), torch.nn.ReLU(),
        ]
        for size in self.layer_sizes[1:-1]:
            layer = [torch.nn.Linear(size[0], size[1]), torch.nn.ReLU()]
            layers.extend(layer)

        layers.append(torch.nn.Linear(self.layer_sizes[-1][0], self.state_space.shape[0]))

        self.body = torch.nn.Sequential(*layers)

    def forward(self, state, action, reward):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(-1).to(self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(-1).to(self.device)
        return self.body(torch.cat([state, action, reward], dim=1))

    def loss(self, state, action, reward, next_state, writer=None, writer_step=None):
        pred_state = self(state, action, reward)
        next_state = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)

        loss = torch.nn.functional.mse_loss(pred_state, next_state)

        if writer:
            writer.add_scalar('training/step_loss', loss.mean(), writer_step)

        return loss

    def train_batch(self, batch, global_steps, writer, gradient_clip):
        state, action, reward, next_state, _ = batch
        loss = self.loss(state, action, reward, next_state, writer, global_steps)
        self.zero_grad()
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)

        if writer and global_steps % 1000000 == 0:
            writer.add_figure('gradient_flow', plot_grad_flow(self.named_parameters()),
                    global_steps)

        self.optimizer.step()

        return loss.detach().cpu().numpy()
