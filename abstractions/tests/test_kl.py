import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import torch

#%%
from dm_control import suite

for env_name, task_name in [
    ('finger', 'spin'),
    ('cartpole', 'swingup'),
    ('cheetah', 'run'),
]:
    env = suite.load(env_name, task_name=task_name)
    print(env_name, task_name, env.action_spec().shape)
print()

#%% Gym
# import gym

# env = gym.make('Humanoid-v2')
# env.reset()
# for i in range(100):
#     env.step(0)
#     env.render()


#%% KL
p = torch.as_tensor([9/25, 12/25, 4/25]).float()
log_p = p.log()
q = torch.as_tensor([1/3, 1/3, 1/3]).float()
log_q = q.log()
kl = torch.nn.functional.kl_div(log_p, q, reduction='sum')
print(kl)
print(sum([-4/3*np.log(2), -2*np.log(3), 2*np.log(5)]))
print()

kl = torch.nn.functional.kl_div(log_q, p, reduction='sum')
print(kl)
print(sum([32/25*np.log(2), 55/25*np.log(3), -50/25*np.log(5)]))
print()

#%%
x = np.linspace(-10,10,num=10000)
dx = x[1]-x[0]
p = stats.norm(loc=0, scale=1).pdf(x)
q = stats.norm(loc=0.3, scale=2.5).pdf(x)

p = torch.as_tensor(p).float()
log_p = p.log()
q = torch.as_tensor(q).float()
log_q = q.log()

dp = torch.distributions.Normal(loc=torch.zeros(1), scale=torch.ones(1))
dq = torch.distributions.Normal(loc=0.3*torch.ones(1), scale=2.5*torch.ones(1))

print(torch.nn.functional.kl_div(log_q, p, reduction='sum')*dx)
print(torch.distributions.kl_divergence(dp, dq))
print()

print(torch.nn.functional.kl_div(log_p, q, reduction='sum')*dx)
print(torch.distributions.kl_divergence(dq, dp))
print()

#%%



#%% Plot stuff
# sns.lineplot(x=x, y=p)
# sns.lineplot(x=x, y=q)
# plt.show()
# %%
