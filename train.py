import gym
import numpy as np
import random
import torch
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args, make_atari, append_timestamp
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

    if args.model_type == 'cnn':
        assert args.num_frames
        env = make_atari(env, args.num_frames)
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

    # Initialize optimizer
    optimizer = torch.optim.Adam(agent.online.parameters(), lr=args.lr)

    # Load checkpoint
    if args.load_checkpoint_path:
        checkpoint = torch.load(args.load_checkpoint_path)
        agent.online.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.online.train()

    # Save path
    if args.model_path:
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)

    # Logging for tensorboard
    if args.output_path:
        writer = SummaryWriter(args.output_path)
    else:
        writer = SummaryWriter(comment=args.env)

    # Episode loop
    global_steps = 0
    episode = 0
    while global_steps < args.max_steps:
        print(f"Episode: {episode}, steps: {global_steps}")
        state = env.reset()
        done = False
        agent.set_epsilon(global_steps, writer)

        cumulative_loss = 0
        steps = 1
        # Collect data from the environment
        while not done:
            global_steps += 1
            action = agent.online.act(state, agent.online.epsilon)
            next_state, reward, done, _ = env.step(action)
            steps += 1
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
            
            if args.model_path:
                if global_steps % args.checkpoint_steps == 0:
                    for filename in os.listdir(f"{args.model_path}/"):
                        if "checkpoint" in filename:
                            os.remove(f"{args.model_path}/" + filename)
                    torch.save(
                        {
                            "global_steps": global_steps,
                            "model_state_dict": agent.online.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss
                        },
                        append_timestamp(f"{args.model_path}/checkpoint_{args.env}")
                        + f"_{global_steps}.tar")

        writer.add_scalar('training/avg_episode_loss', cumulative_loss / steps,
                          episode)
        episode += 1
        if len(agent.replay_buffer
              ) < args.batchsize or global_steps < args.warmup_period:
            continue

        # Testing policy
        if episode % args.test_policy_episodes == 0:
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
                    action = agent.online.act(state,
                                              0)  # passing in epsilon = 0

                    # Update reward
                    cumulative_reward += reward

                env.close()  # close viewer

                print(f"Policy_reward for test: {cumulative_reward}")

                # Logging
                writer.add_scalar('validation/policy_reward', cumulative_reward,
                                  episode)

    env.close()
    if args.model_path:
        torch.save(agent.online,
                   append_timestamp(f"{args.model_path}/{args.env}") + ".pth")

