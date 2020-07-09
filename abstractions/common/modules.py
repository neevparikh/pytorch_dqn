import torch

from .utils import weights_init_, conv2d_size_out


class Reshape(torch.nn.Module):
    """Module that returns a view of the input which has a different size

    Parameters
    ----------
    args : int...
        The desired size
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def __repr__(self):
        s = self.__class__.__name__
        s += '{}'.format(self.shape)
        return s

    def forward(self, input):
        return input.view(*self.shape)

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes, activation=torch.nn.ReLU, final_activation=None):
        super(MLP, self).__init__()
        layer_shapes = list(zip(layer_sizes[:-1],layer_sizes[1:]))
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

class InverseModel(torch.nn.Module):
    def __init__(self, args, feature_size, num_actions):
        super(InverseModel, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(feature_size * 2, args.hidden_size),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(args.hidden_size, num_actions))

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)


class ContrastiveModel(torch.nn.Module):
    def __init__(self, args, feature_size):
        super(ContrastiveModel, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(feature_size * 2, args.hidden_size),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(args.hidden_size, 1))

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)


class MarkovHead(torch.nn.Module):
    def __init__(self, args, feature_size, num_actions):
        super(MarkovHead, self).__init__()
        self.inverse_model = InverseModel(args, feature_size, num_actions)
        self.discriminator = ContrastiveModel(args, feature_size)

        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_markov_loss(self, z0, z1, a):
        # Inverse loss
        log_pr_actions = self.inverse_model(z0, z1)
        l_inverse = self.ce(input=log_pr_actions, target=a)

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
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = torch.nn.functional.relu(self.linear1(xu))
        x1 = torch.nn.functional.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.nn.functional.relu(self.linear4(xu))
        x2 = torch.nn.functional.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

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

    def forward(self, state):
        x = torch.nn.functional.relu(self.linear1(state))
        x = torch.nn.functional.relu(self.linear2(x))
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
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = torch.nn.Linear(num_inputs, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

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

    def forward(self, state):
        x = torch.nn.functional.relu(self.linear1(state))
        x = torch.nn.functional.relu(self.linear2(x))
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
