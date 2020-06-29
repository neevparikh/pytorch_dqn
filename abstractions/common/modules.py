import torch


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
