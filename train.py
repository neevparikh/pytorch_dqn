import gym
import numpy as np
import random
import torch
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args, get_state_on_step, get_state_on_reset
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
        env = args.env

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
        "epsilon_decay_end": args.epsilon_decay_end,
        "warmup_period": args.warmup_period,
        "double_DQN": not (args.vanilla_DQN),
        "model_type": args.model_type,
        "num_frames": args.num_frames,
    }
    agent = DQN_agent(**agent_args)

    if args.model_type == 'cnn' or args.model_type == 'cnn_render':
        assert args.num_frames
        last_num_frames = deque(maxlen=args.num_frames)
    else:
        last_num_frames = None

    # Initialize optimizer
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=args.lr)

    # Logging for tensorboard
    if args.output_path:
        writer = SummaryWriter(args.output_path)
    else:
        writer = SummaryWriter(comment=args.env)

    # Episode loop
    global_steps = 0
    episode = 0
    while global_steps < args.max_steps:
        state, prev_frame = get_state_on_reset(env, args.model_type,
                                               last_num_frames, args.num_frames)
        done = False
        agent.set_epsilon(global_steps, writer)

        cumulative_loss = 0
        step = 1
        # Collect data from the environment
        while not done:
            global_steps += 1
            action = agent.online.act(state, agent.online.epsilon)
            next_state, reward, done, _, prev_frame = get_state_on_step(
                env, args.model_type, action, last_num_frames, args.num_frames,
                prev_frame)
            if args.reward_clip:
                clipped_reward = np.clip(reward, -args.reward_clip,
                                         args.reward_clip)
            else:
                clipped_reward = reward
            agent.replay_buffer.append(
                Experience(state, action, clipped_reward, next_state,
                           int(done)))
            state = next_state

            # If not enough data, try again
            if len(agent.replay_buffer
                  ) < args.batchsize or global_steps < args.warmup_period:
                continue

            # Training loop
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

            # Average over update steps
            cumulative_loss /= args.update_steps
            step += 1

        writer.add_scalar('training/avg_episode_loss', cumulative_loss / step,
                          episode)

        if len(agent.replay_buffer
              ) < args.batchsize or global_steps < args.warmup_period:
            episode += 1
            continue
        # Testing policy
        with torch.no_grad():
            # Reset environment
            cumulative_reward = 0
            state, prev_frame = get_state_on_reset(env, args.model_type,
                                                   last_num_frames,
                                                   args.num_frames)
            action = agent.online.act(state, 0)
            done = False
            render = args.render and (episode % args.render_episodes == 0)

            # Test episode loop
            while not done:
                # Take action in env
                if render:
                    env.render()

                state, reward, done, _, prev_frame = get_state_on_step(
                    env, args.model_type, action, last_num_frames,
                    args.num_frames, prev_frame)
                action = agent.online.act(state, 0)  # passing in epsilon = 0

                # Update reward
                cumulative_reward += reward

            env.close()  # close viewer

            print(f"Episode: {episode}, steps: {global_steps}, policy_reward: {cumulative_reward}")

            # Logging
            writer.add_scalar('validation/policy_reward', cumulative_reward,
                              episode)
            episode += 1

    env.close()
    if args.model_path:
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)
        torch.save(agent.online, f"{args.model_path}/latest.pth")
