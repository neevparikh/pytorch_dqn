# Adapted from https://github.com/pranz24/pytorch-soft-actor-critic (2020)
import time
import os
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .model import SAC
from ...common.replay_buffer import ReplayBuffer
from ...common.utils import initialize_environment, reset_seeds
from ...common.parsers import sac_parser


def test_policy(test_env, agent, episode, global_steps, writer, log_filename, args):
    with torch.no_grad():
        # Reset environment
        cumulative_reward = 0

        for _ in range(args.episodes_per_eval):
            test_state = test_env.reset()

            test_done = False
            render = args.render and (episode % args.render_episodes == 0)

            # Test episode loop
            while not test_done:
                test_action = agent.act(test_state, evaluate=True)

                # Take action in env
                if render:
                    test_env.render()

                test_state, test_reward, test_done, _ = test_env.step(test_action)

                # Update reward
                cumulative_reward += test_reward

        eval_reward = cumulative_reward/args.episodes_per_eval

        print("Policy_reward for test:", eval_reward)

        # Logging
        if not args.no_tensorboard:
            writer.add_scalar('validation/policy_reward', eval_reward, global_steps)
        if log_filename:
            with open(log_filename, "a") as f:
                f.write("{},{},{},".format(episode, global_steps, eval_reward))


def episode_loop(env, test_env, agent, replay_buffer, args, writer):
    # Episode loop
    global_steps = 0
    steps = 1
    episode = 0
    updates = 0
    start = time.time()
    t_zero = time.time()
                 
    end = time.time() + 1

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

        steps = 1
        score = 0

        # Collect data from the environment
        while not done:
            if global_steps <= args.warmup_period:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            
            if len(replay_buffer) >= args.batchsize and global_steps > args.warmup_period:
                for _ in range(args.updates_per_step):
                    # Update parameters of all the networks
                    result = agent.update_parameters(replay_buffer, args.batchsize, updates)
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = result

                    if writer:
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            

            next_state, reward, done, _ = env.step(action)
            score += reward
            steps += 1
            if args.reward_clip:
                clipped_reward = np.clip(reward, -args.reward_clip, args.reward_clip)
            else:
                clipped_reward = reward

            # mask = 1 if steps == env._max_episode_steps else not done
            # pylint: disable=protected-access
            mask = not done

            # Store in replay buffer
            replay_buffer.append(state, action, clipped_reward, next_state, int(mask))
            state = next_state

            # Testing policy
            if global_steps % args.test_policy_steps == 0:
                test_policy(test_env, agent, episode, global_steps, writer, log_filename, args)
                if log_filename:
                    with open(log_filename, "a") as f:
                        f.write("{:.2f}\n".format(time.time() - t_zero))

            global_steps += 1

        end = time.time()

        episode += 1


args = sac_parser.parse_args()

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
agent = SAC(env.observation_space.shape, env.action_space, device, args)

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

replay_buffer = ReplayBuffer(args.replay_buffer_size)
episode_loop(env, test_env, agent, replay_buffer, args, writer)

env.close()
test_env.close()

# if args.model_path:
#     torch.save(agent.online, append_timestamp(os.path.join(args.model_path, args.run_tag)) + ".pth")
