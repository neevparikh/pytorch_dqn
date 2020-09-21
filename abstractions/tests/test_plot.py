# import matplotlib
# matplotlib.use('macosx')
import matplotlib.pyplot as plt

# Importing dm_control.suite or instantiating a dm2gym environment
# causes plt.show() to crash. A fix for this bug is to switch the
# matplotlib backend (see above), but this no longer waits for the
# user to manually close any open plots.

# from dm_control import suite

# import gym
# env = gym.make('dm2gym:CartpoleSwingup-v0')

plt.plot()
plt.show()
