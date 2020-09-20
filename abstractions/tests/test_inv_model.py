import numpy as np
from scipy import stats
import seaborn as sns
from sys import platform as sys_pf
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import torch
from dm_control import suite
from tqdm import tqdm

from ..common.modules import MarkovHead
from ..common.utils import initialize_environment, reset_seeds
from ..common.parsers import sac_parser

def main():
    args = sac_parser.parse_args()

    env_name, task_name = 'cartpole', 'swingup'
    # env_name, task_name = 'finger', 'spin',
        # 'cheetah', 'run',
    env = suite.load(env_name, task_name=task_name)
    print(env_name, task_name, env.action_spec().shape)


    # Set seeds
    reset_seeds(args.seed)

    # Initialize envs
    # env, test_env = initialize_environment(args)
    batch_size = 128
    n_features = 20
    n_action_dims = env.action_spec().shape[0]
    markov_head = MarkovHead(args, n_features, n_action_dims)
    z0 = torch.randn(1, n_features).expand(batch_size, n_features)
    z1 = torch.randn(1, n_features).expand(batch_size, n_features)
    a = 0.5*torch.randn(batch_size, n_action_dims)+2.5
    optimizer = torch.optim.Adam(markov_head.parameters())
    losses = []
    for _ in tqdm(range(1000)):
        loss = markov_head.compute_markov_loss(z0, z1, a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = loss.detach().item()
        # print(x)
        losses.append(x)

    plt.plot(np.arange(len(losses)), np.asarray(losses))
    plt.savefig('test_inv_model.png')
    plt.show()

if __name__ == "__main__":
    main()