import gym
import numpy as np
import random
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args, plot_grad_flow
from model import DQN_agent, Experience

if __name__ == "__main__":
    args = parse_args()

    # Setting cuda seeds
    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

    # Setting random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize environment
    if type(args.env) == str:
        env = gym.make(args.env)
    else:
        env = env

    if type(env.action_space) != gym.spaces.Discrete:
        raise NotImplementedError("DQN for continuous action_spaces hasn't been\
                implemented")

    # Check if GPU can be used and was asked for
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Initialize model
    agent_args = {
        "device": device,
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "num_actions": env.action_space.n,
        "target_moving_average": args.target_moving_average,
        "gamma": args.gamma,
        "replay_buffer_size": args.replay_buffer_size,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_decay_start": args.epsilon_decay_start,
        "double_DQN": not(args.vanilla_DQN),
    }
    agent = DQN_agent(**agent_args)

    # Initialize optimizer
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=args.lr)

    # Logging for tensorboard
    if args.output_path:
        writer = SummaryWriter(args.output_path)
    else:
        writer = SummaryWriter(comment=args.env)

    # Episode loop
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        agent.set_epsilon(episode, writer)

        # Collect data from the environment
        with torch.no_grad():
            while not done:
                action = agent.online.act(state, agent.online.epsilon)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.append(
                    Experience(
                        state, action,
                        np.clip(reward, -args.reward_clip, args.reward_clip),
                        next_state, int(done)))
                state = next_state

        # If not enough data, try again
        if len(agent.replay_buffer) < args.batchsize:
            continue

        # Training loop
        cumulative_loss = 0
        for i in range(args.update_steps):
            # This is list<experiences>
            minibatch = random.sample(agent.replay_buffer, args.batchsize)

            # This is experience<list<states>, list<actions>, ...>
            minibatch = Experience(*zip(*minibatch))
            optimizer.zero_grad()

            # Get loss
            loss = agent.loss_func(minibatch, writer, episode)

            cumulative_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(),
                                           args.gradient_clip)
            # Update parameters
            optimizer.step()
            agent.sync_networks()

        # Gradient flow in network
        writer.add_figure('rbf_training/gradient_flow',
                          plot_grad_flow(agent.online.named_parameters(),
                                         episode),
                          global_step=episode)

        # Testing policy
        with torch.no_grad():
            # Reset environment
            cumulative_reward = 0
            state = env.reset()
            action = agent.online.act(state, 0)
            done = False
            render = args.render and (episode % args.render_episodes == 0)

            # Test episode loop
            while not done:
                # Take action in env
                if render:
                    env.render()
                state, reward, done, _ = env.step(action)
                action = agent.online.act(state, 0)  # passing in epsilon = 0

                # Update reward
                cumulative_reward += reward

            env.close()  # close viewer

            print(f"Episode: {episode}, policy_reward: {cumulative_reward}")

            # Logging
            writer.add_scalar('training/episode_loss',
                              cumulative_loss / args.update_steps, episode)
            writer.add_scalar('validation/policy_reward', cumulative_reward,
                              episode)

    env.close()
