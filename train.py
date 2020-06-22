import random
import time
import os
import json

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args, make_ari, make_atari, append_timestamp, reset_seeds, \
        initialize_environment
from model import DQN_agent


def test_policy(test_env, agent, episode, global_steps, writer, log_filename, args):
    with torch.no_grad():
        # Reset environment
        cumulative_reward = 0

        test_state = test_env.reset()

        test_action = agent.online.act(test_state, 0)
        test_done = False
        render = args.render and (episode % args.render_episodes == 0)

        # Test episode loop
        while not test_done:
            # Take action in env
            if render:
                test_env.render()

            test_state, test_reward, test_done, _ = test_env.step(test_action)

            # passing in epsilon = 0
            test_action = agent.online.act(test_state, 0)

            # Update reward
            cumulative_reward += test_reward

        print("Policy_reward for test:", cumulative_reward)

        # Logging
        if not args.no_tensorboard:
            writer.add_scalar('validation/policy_reward', cumulative_reward, global_steps)
        if log_filename:
            with open(log_filename, "a") as f:
                f.write("{},{},{},".format(episode, global_steps, cumulative_reward))


def episode_loop(env, test_env, agent, args, writer):
    # Episode loop
    global_steps = 0
    steps = 1
    episode = 0
    start = time.time()
    t_zero = time.time()
                 
    end = time.time() + 1

    if args.load_checkpoint_path:
        checkpoint = agent.load_checkpoint(args.load_checkpoint_path)
        global_steps = checkpoint['global_steps']
        episode = checkpoint['episode']

    score = 0
    while global_steps < args.max_steps:
        info_str = "episode: {}, ".format(episode)
        info_str += "steps: {}, ".format(global_steps)
        info_str += "ep_score: {}, ".format(score)
        info_str += "FPS: {}".format(steps/(end - start))
        print(info_str)
        start = time.time()

        state = env.reset()
        done = False
        agent.set_epsilon(global_steps, writer)

        cumulative_loss = 0
        steps = 1
        score = 0
        # Collect data from the environment
        while not done:
            global_steps += 1
            action = agent.online.act(state, agent.online.epsilon)

            next_state, reward, done, info = env.step(action)
            score += reward
            steps += 1
            if args.reward_clip:
                clipped_reward = np.clip(reward, -args.reward_clip, args.reward_clip)
            else:
                clipped_reward = reward

            # Store in replay buffer
            agent.replay_buffer.append(state, action, clipped_reward, next_state, int(done))
            state = next_state

            # If not enough data, try again
            if len(agent.replay_buffer) < args.batchsize or global_steps < args.warmup_period:
                continue

            # Training loop
            loss = agent.train_batch(args.batchsize, global_steps, writer, args.gradient_clip)
            cumulative_loss += loss.item()

            if args.model_path is not None and global_steps % args.checkpoint_steps == 0:
                agent.save_checkpoint(episode, global_steps, args)

            # Testing policy
            if global_steps % args.test_policy_steps == 0:
                test_policy(test_env, agent, episode, global_steps, writer, log_filename, args)
                if log_filename:
                    with open(log_filename, "a") as f:
                        f.write("{:.2f}\n".format(time.time() - t_zero))

        if not args.no_tensorboard:
            writer.add_scalar('training/avg_episode_loss', cumulative_loss / steps, episode)
        end = time.time()

        episode += 1


args = parse_args()

# Set seeds
reset_seeds(args.seed)

# Initialize envs
env, test_env = initialize_environment(args)

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
    "lr": args.lr,
    "target_moving_average": args.target_moving_average,
    "gamma": args.gamma,
    "replay_buffer_size": args.replay_buffer_size,
    "epsilon_decay_length": args.epsilon_decay_length,
    "final_epsilon_value": args.final_epsilon_value,
    "warmup_period": args.warmup_period,
    "double_DQN": not (args.vanilla_DQN),
    "model_type": args.model_type,
    "model_shape": args.model_shape,
    "num_frames": args.num_frames,
}
agent = DQN_agent(**agent_args)

# Save path
if args.model_path:
    os.makedirs(args.model_path, exist_ok=True)

if args.output_path:
    os.makedirs(args.output_path, exist_ok=True)

# Logging via csv
if args.output_path:
    base_filename = os.path.join(args.output_path, args.run_tag)
    os.makedirs(base_filename, exist_ok=True)
    log_filename = os.path.join(base_filename, 'reward.csv')
    with open(log_filename, "w") as f:
        f.write("episode,steps,reward,runtime\n")
    with open(os.path.join(base_filename, 'params.json'), 'w') as fp:
        param_dict = vars(args).copy()
        del param_dict['output_path']
        del param_dict['model_path']
        json.dump(param_dict, fp)
else:
    log_filename = None

# Logging for tensorboard
if not args.no_tensorboard:
    writer = SummaryWriter(comment=args.run_tag)
else:
    writer = None

episode_loop(env, test_env, agent, args, writer)

env.close()
test_env.close()

if args.model_path:
    torch.save(agent.online, append_timestamp(os.path.join(args.model_path, args.run_tag)) + ".pth")
