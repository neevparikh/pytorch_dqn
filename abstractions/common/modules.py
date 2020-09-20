import torch

from .utils import weights_init_, conv2d_size_out


class Reshape(torch.nn.Module):
    """
    Description:
        Module that returns a view of the input which has a different size

    Parameters:
        - args : Int...
            The desired size
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def __repr__(self):
        s = self.__class__.__name__
        s += '{}'.format(self.shape)
        return s

    def forward(self, x):
        return x.view(*self.shape)


## DQN ##


class MLP(torch.nn.Module):
    """
    Description:
        Multilayer perceptron module for plugging into other implementations

    Parameters:
        - layer_sizes : List[Int]
            The layer sizes as int, i.e. [10, 15, 15, 5]
        - activation : func
            Function for activation
        - final_activation : func
            Function for activation of the last layer
    """
    def __init__(self, layer_sizes, activation=torch.nn.ReLU, final_activation=None):
        super(MLP, self).__init__()
        layer_shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        layers = []
        for shape in layer_shapes[:-1]:
            layers.append(torch.nn.Linear(*shape))
            if activation is not None:
                layers.append(activation())
        layers.append(torch.nn.Linear(*layer_shapes[-1]))
        if final_activation is not None:
            layers.append(final_activation())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


## Markov Abstraction ##


def build_phi_network(args, input_shape):
    """
    Description:
        Construct the appropriate kind of phi network for the pretraining step in the markov
        abstractions

    Parameters:
        - args : Namespace
            See the `markov` folder for more information on the argparse
        - input_shape : Tuple[Int]
            Shape of the input (state generally)
    """
    if args.model_type == 'cnn':
        final_size = conv2d_size_out(input_shape, (8, 8), 4)
        final_size = conv2d_size_out(final_size, (4, 4), 2)
        final_size = conv2d_size_out(final_size, (3, 3), 1)
        output_size = final_size[0] * final_size[1] * 64
        phi = torch.nn.Sequential(*[
            torch.nn.Conv2d(args.num_frames, 32, kernel_size=(8, 8), stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            Reshape(-1, output_size),
        ])
    elif args.model_type == 'mlp':
        if args.model_shape == 'small':
            layer_sizes = [(64, 64), (64, 64)]
        elif args.model_shape == 'medium':
            layer_sizes = [(256, 256), (256, 256), (256, 256)]
        elif args.model_shape == 'large':
            layer_sizes = [(1024, 1024), (1024, 1024), (1024, 1024), (1024, 1024)]
        elif args.model_shape == 'giant':
            layer_sizes = [(2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048), (2048, 2048),
                           (2048, 2048)]
        else:
            raise ValueError("Unrecognized model_shape: ", args.model_shape)
        layers = [torch.nn.Linear(input_shape[0], layer_sizes[0][0]), torch.nn.ReLU()]
        for size in layer_sizes:
            layer = [torch.nn.Linear(size[0], size[1]), torch.nn.ReLU()]
            layers.extend(layer)
        output_size = layer_sizes[-1][1]
    else:
        raise ValueError("Unrecognized model_type: ", args.model_type)

    return phi, output_size


class InverseModel(torch.nn.Module):
    """
    Description:
        Network module that captures predicting the action given a state, next_state pair.

    Parameters:
        - args : Namespace
            See the `markov` folder for information on the argparse.
        - feature_size : Int
            The size of the abstract state
        - num_actions : Int
            The number of actions in the environment

    """
    def __init__(self, args, feature_size, num_actions, discrete=False):
        super(InverseModel, self).__init__()
        self.discrete = discrete
        self.body = torch.nn.Sequential(
            torch.nn.Linear(feature_size * 2, args.hidden_size),
            torch.nn.ReLU(),
        )
        if self.discrete:
            self.log_pr_linear = torch.nn.Linear(args.hidden_size, num_actions)
        else:
            self.mean_linear = torch.nn.Linear(args.hidden_size, num_actions)
            self.log_std_linear = torch.nn.Linear(args.hidden_size, num_actions)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        shared_vector = self.body(context)

        if self.discrete:
            return self.log_pr_linear(shared_vector)
        else:
            mean = self.mean_linear(shared_vector)
            log_std = self.log_std_linear(shared_vector)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std = log_std.exp()
            return mean, std


class ContrastiveModel(torch.nn.Module):
    """
    Description:
        Network module that captures if a given state1, state2 pair belong in the same transition.

    Parameters:
        - args : Namespace
            See the `markov` folder for information on the argparse.
        - feature_size : Int
            The size of the abstract state
    """
    def __init__(self, args, feature_size):
        super(ContrastiveModel, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(feature_size * 2, args.hidden_size),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(args.hidden_size, 1))

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)


class MarkovHead(torch.nn.Module):
    """
    Description:
        Network module that combines contrastive and inverse models.

    Parameters:
        - args : Namespace
            See the `markov` folder for information on the argparse.
        - feature_size : Int
            The size of the abstract state
        - num_actions : Int
            The number of actions in the environment

    Notes:
        - This does not support continuous action spaces right now. TODO
    """
    def __init__(self, args, feature_size, num_actions):
        super(MarkovHead, self).__init__()
        self.discrete = hasattr(args, 'discrete') and args.discrete
        self.n_actions = num_actions

        self.inverse_model = InverseModel(args, feature_size, num_actions, discrete=self.discrete)
        self.discriminator = ContrastiveModel(args, feature_size)

        self.bce = torch.nn.BCEWithLogitsLoss()
        if self.discrete:
            self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_markov_loss(self, z0, z1, a):
        # Inverse loss
        if self.discrete:
            log_pr_actions = self.inverse_model(z0, z1)
            l_inverse = self.ce(input=log_pr_actions, target=a)
        else:
            mean, std = self.inverse_model(z0, z1)
            cov = torch.diag_embed(std, dim1=1, dim2=2)
            normal = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
            log_pr_action = normal.log_prob(a)
            l_inverse = -1*log_pr_action.mean(dim=0)

        # Ratio loss
        with torch.no_grad():
            N = len(z1)
            idx = torch.randperm(N)  # shuffle indices of next states
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_real_transition = torch.cat([torch.ones(N), torch.zeros(N)], dim=0).to(z0.device)
        log_pr_real = self.discriminator(z0_extended, z1_pos_neg)
        l_ratio = self.bce(input=log_pr_real, target=is_real_transition.unsqueeze(-1).float())

        markov_loss = l_inverse + l_ratio
        return markov_loss


## SAC ##

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
SAC_epsilon = 1e-6


class ValueNetwork(torch.nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = torch.nn.functional.relu(self.linear1(state))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(torch.nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim, model_type, num_frames=None):
        super(QNetwork, self).__init__()

        if len(input_shape) > 2:
            input_shape = input_shape[1:]

        self.model_type = model_type
        # Q1 architecture
        if model_type == 'mlp':
            assert len(input_shape) == 1, "Cannot use {} as input".format(input_shape)
            self.q1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0] + num_actions, hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, 1))
            self.q2 = torch.nn.Sequential(torch.nn.Linear(input_shape[0] + num_actions, hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, 1))
        elif model_type == 'cnn':
            assert num_frames is not None, "Must provide num frames"

            final_size = conv2d_size_out(input_shape, (8, 8), 4)
            final_size = conv2d_size_out(final_size, (4, 4), 2)
            final_size = conv2d_size_out(final_size, (3, 3), 1)
            output_size = final_size[0] * final_size[1] * 64

            self.body1 = torch.nn.Sequential(
                torch.nn.Conv2d(num_frames, 32, kernel_size=(8, 8), stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                torch.nn.ReLU(),
                Reshape(-1, output_size))
            self.q1 = torch.nn.Sequential(torch.nn.Linear(output_size + num_actions, hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, 1))

            self.body2 = torch.nn.Sequential(
                torch.nn.Conv2d(num_frames, 32, kernel_size=(8, 8), stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                torch.nn.ReLU(),
                Reshape(-1, output_size))
            self.q2 = torch.nn.Sequential(torch.nn.Linear(output_size + num_actions, hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_dim, 1))

        self.apply(weights_init_)

        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(repr(self))
        print("Number of trainable parameters: {}".format(trainable_parameters))

    def forward(self, state, action):
        if self.model_type == 'cnn':
            x1 = self.q1(torch.cat([self.body1(state), action], 1))
            x2 = self.q2(torch.cat([self.body2(state), action], 1))
        elif self.model_type == 'mlp':
            xu = torch.cat([state, action], 1)
            x1 = self.q1(xu)
            x2 = self.q2(xu)
        return x1, x2


class GaussianPolicy(torch.nn.Module):
    def __init__(self,
                 input_shape,
                 num_actions,
                 hidden_dim,
                 model_type,
                 num_frames=None,
                 action_space=None):
        super(GaussianPolicy, self).__init__()

        if len(input_shape) > 2:
            input_shape = input_shape[1:]

        if model_type == 'mlp':
            assert len(input_shape) == 1, "Cannot use {} as input".format(input_shape)
            self.body = torch.nn.Sequential(torch.nn.Linear(input_shape[0], hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim),
                                            torch.nn.ReLU())

        elif model_type == 'cnn':
            assert num_frames is not None, "Must provide num frames"

            final_size = conv2d_size_out(input_shape, (8, 8), 4)
            final_size = conv2d_size_out(final_size, (4, 4), 2)
            final_size = conv2d_size_out(final_size, (3, 3), 1)
            output_size = final_size[0] * final_size[1] * 64

            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(num_frames, 32, kernel_size=(8, 8), stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                torch.nn.ReLU(),
                Reshape(-1, output_size),
                torch.nn.Linear(output_size, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU())

        self.mean_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = torch.nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(repr(self))
        print("Number of trainable parameters: {}".format(trainable_parameters))

    def forward(self, state):
        x = self.body(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + SAC_epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(torch.nn.Module):
    def __init__(self,
                 input_shape,
                 num_actions,
                 hidden_dim,
                 model_type,
                 num_frames=None,
                 action_space=None):
        super(DeterministicPolicy, self).__init__()

        if len(input_shape) > 2:
            input_shape = input_shape[1:]

        if model_type == 'mlp':
            assert len(input_shape) == 1, "Cannot use {} as input".format(input_shape)
            self.body = torch.nn.Sequential(torch.nn.Linear(input_shape[0], hidden_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(hidden_dim, hidden_dim),
                                            torch.nn.ReLU())

        elif model_type == 'cnn':
            assert num_frames is not None, "Must provide num frames"

            final_size = conv2d_size_out(input_shape, (8, 8), 4)
            final_size = conv2d_size_out(final_size, (4, 4), 2)
            final_size = conv2d_size_out(final_size, (3, 3), 1)
            output_size = final_size[0] * final_size[1] * 64

            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(num_frames, 32, kernel_size=(8, 8), stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
                torch.nn.ReLU(),
                Reshape(-1, output_size),
                torch.nn.Linear(output_size, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU())

        self.mean = torch.nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(repr(self))
        print("Number of trainable parameters: {}".format(trainable_parameters))

    def forward(self, state):
        x = self.body(state)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


## PPO ##


# Continuous
class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, discrete, device, **kwargs):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.discrete = discrete
        self.device = device
        if discrete:
            assert 'hidden_size' in kwargs.keys(), "Must provide hidden_size"
            hidden_size = kwargs['hidden_size']
            # actor
            self.actor = torch.nn.Sequential(torch.nn.Linear(state_dim, hidden_size),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(hidden_size, hidden_size),
                                             torch.nn.Tanh(),
                                             torch.nn.Linear(hidden_size, action_dim),
                                             torch.nn.Softmax(dim=-1))

            # critic
            self.critic = torch.nn.Sequential(torch.nn.Linear(state_dim, hidden_size),
                                              torch.nn.Tanh(),
                                              torch.nn.Linear(hidden_size, hidden_size),
                                              torch.nn.Tanh(),
                                              torch.nn.Linear(hidden_size, 1))
        else:
            assert 'action_std' in kwargs.keys(), "Must provide action_std"
            action_std = kwargs['action_std']
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, action_dim),
                torch.nn.Tanh(),
            )
            # critic
            self.critic = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 1),
            )
            self.action_var = torch.full((action_dim,), action_std**2).to(device)

        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(repr(self))
        print("Number of trainable parameters: {}".format(trainable_parameters))

    def forward(self):
        raise NotImplementedError

    def act(self, state, rollouts):
        if self.discrete:
            state = torch.from_numpy(state).float().to(self.device)
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            rollouts.states.append(state)
            rollouts.actions.append(action)
            rollouts.logprobs.append(dist.log_prob(action))

            return action.item()
        else:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).to(self.device)

            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

            rollouts.states.append(state)
            rollouts.actions.append(action)
            rollouts.logprobs.append(action_logprob)

            return action.detach()

    def evaluate(self, state, action):
        if self.discrete:
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            state_value = self.critic(state)

            return action_logprobs, torch.squeeze(state_value), dist_entropy
        else:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)

            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_value = self.critic(state)

            return action_logprobs, torch.squeeze(state_value), dist_entropy
