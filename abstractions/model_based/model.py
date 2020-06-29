import torch

from ..common.modules import MLP

class ModelNet(torch.nn.Module):
    def __init__(self, args, device, state_space, action_space):
        super(ModelNet, self).__init__()
        self.state_space = state_space
        self.num_actions = action_space.n
        self.action_space = action_space
        self.device = device

        if args.model_shape == 'tiny':
            self.layer_sizes = [15]
        elif args.model_shape == 'small':
            self.layer_sizes = [16]*4
        elif args.model_shape == 'medium':
            self.layer_sizes = [128]*4
        elif args.model_shape == 'large':
            self.layer_sizes = [1024]*5
        elif args.model_shape == 'giant':
            self.layer_sizes = [2048]*7

        self.build_model()

        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def build_model(self):
        input_size = self.state_space.shape[0] + 1
        self.body = MLP([input_size] + self.layer_sizes,
                        activation=torch.nn.ReLU,
                        final_activation=torch.nn.ReLU
        )

        self.state_head = torch.nn.Linear(self.layer_sizes[-1], self.state_space.shape[0])
        self.reward_head = torch.nn.Linear(self.state_space.shape[0] * 2 + 1, 1)
        self.done_head = torch.nn.Sequential(torch.nn.Linear(self.state_space.shape[0] * 2 + 1, 1),
            torch.nn.Sigmoid())

    def forward(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action = torch.as_tensor(action, dtype=torch.float32).unsqueeze(-1).to(self.device)
        output = self.body(torch.cat([state, action], dim=1))
        pred_state = self.state_head(output)
        sasp_input = torch.cat([state, pred_state, action], dim=1)
        return pred_state, self.reward_head(sasp_input), self.done_head(sasp_input)

    def loss(self, batch, writer=None, writer_step=None):
        state, action, reward, next_state, done = batch
        pred_state, pred_reward, pred_done = self(state, action)

        next_state = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(-1).to(self.device)
        done = torch.as_tensor(done, dtype=torch.float32).unsqueeze(-1).to(self.device)

        state_loss = torch.nn.functional.mse_loss(pred_state, next_state) 
        reward_loss = torch.nn.functional.mse_loss(pred_reward, reward)
        done_loss = torch.nn.functional.binary_cross_entropy(pred_done, done)

        loss = state_loss + reward_loss + done_loss

        if writer:
            writer.add_scalar('stepwise/reward_loss', reward_loss.mean(), writer_step)
            writer.add_scalar('stepwise/done_loss', done_loss.mean(), writer_step)
            writer.add_scalar('stepwise/state_loss', state_loss.mean(), writer_step)
            writer.add_scalar('stepwise/step_loss', loss.mean(), writer_step)

        if writer and writer_step % 5000 == 0:
            writer.add_text('environment/next_state', str(next_state), writer_step)
            writer.add_text('environment/reward', str(reward), writer_step)
            writer.add_text('environment/done', str(done), writer_step)

            writer.add_text('predicted/next_state', str(pred_state), writer_step)
            writer.add_text('predicted/reward', str(pred_reward), writer_step)
            writer.add_text('predicted/done', str(pred_done), writer_step)

        return loss

    def train_batch(self, batch, global_steps, writer, gradient_clip):
        loss = self.loss(batch, writer, global_steps)
        self.zero_grad()
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)

        self.optimizer.step()

        return loss.detach().cpu().numpy()
