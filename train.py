import gym
import time
import numpy as np
import random
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from utils import parse_args, make_ari, make_atari, append_timestamp
from model import DQN_agent
from replay_buffer import Experience

args = parse_args()

# Initialize environment
if type(args.env) == str:
    env = gym.make(args.env)
    test_env = gym.make(args.env)
else:
    env = args.env
    test_env = args.env

# Set tag for this run
run_tag = args.run_tag

# Setting cuda seeds
if torch.cuda.is_available():
    torch.backends.cuda.deterministic = True
    torch.backends.cuda.benchmark = False

# Setting random seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
test_env.seed(args.seed)

if args.model_type == 'cnn':
    assert args.num_frames
    if not args.no_atari:
        print("Using atari preprocessing")
        env = make_atari(env, args.num_frames)
        test_env = make_atari(test_env, args.num_frames)

if args.ari:
    print("Using ARI")
    env = make_ari(env)
    test_env = make_ari(test_env)

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
    agent.target.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.online.train()

# Save path
if args.model_path:
    os.makedirs(args.model_path, exist_ok=True)

if args.output_path:
    os.makedirs(args.output_path, exist_ok=True)

# Logging via csv
if args.output_path:
    log_filename = f"{args.output_path}/{run_tag}.csv"
    with open(log_filename, "w") as f:
        f.write("episode,global_steps,cumulative_reward,\n")
else:
    log_filename = None

# Logging for tensorboard
if not args.no_tensorboard:
    writer = SummaryWriter(comment=run_tag)
else:
    writer = None

# Episode loop
global_steps = 0
steps = 1
episode = 0
start = time.time()
end = time.time() + 1

if args.load_checkpoint_path and checkpoint is not None:
    global_steps = checkpoint['global_steps']
    episode = checkpoint['episode']

while global_steps < args.max_steps:
    print(
        f"Episode: {episode}, steps: {global_steps}, FPS: {steps/(end - start)}"
    )
    start = time.time()
    state = env.reset()

    done = False
    agent.set_epsilon(global_steps, writer)

    cumulative_loss = 0
    steps = 1
    # Collect data from the environment
    while not done:
        global_steps += 1
        action = agent.online.act(state, agent.online.epsilon)

        next_state, reward, done, info = env.step(action)

        steps += 1
        if args.reward_clip:
            clipped_reward = np.clip(reward, -args.reward_clip,
                                     args.reward_clip)
        else:
            clipped_reward = reward
        agent.replay_buffer.append(state, action, clipped_reward, next_state,
                                   int(done))
        state = next_state

        # If not enough data, try again
        if len(agent.replay_buffer
              ) < args.batchsize or global_steps < args.warmup_period:
            continue

        # Training loop
        minibatch = agent.replay_buffer.sample(args.batchsize)

        minibatch = Experience(*minibatch)
        optimizer.zero_grad()

        # Get loss
        loss = agent.loss_func(minibatch, writer, global_steps)

        cumulative_loss += loss.item()
        loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(),
                                           args.gradient_clip)
        # Update parameters
        optimizer.step()
        agent.sync_networks()

        if args.model_path:
            if global_steps % args.checkpoint_steps == 0:
                for filename in os.listdir(f"{args.model_path}/"):
                    if "checkpoint" in filename and args.env in filename:
                        os.remove(f"{args.model_path}/" + filename)
                torch.save(
                    {
                        "global_steps": global_steps,
                        "model_state_dict": agent.online.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "episode": episode,
                    },
                    append_timestamp(f"{args.model_path}/checkpoint_{run_tag}")
                    + f"_{global_steps}.tar")

        # Testing policy
        if global_steps % args.test_policy_steps == 0:
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

                    test_state, test_reward, test_done, _ = test_env.step(
                        test_action)

                    # passing in epsilon = 0
                    test_action = agent.online.act(test_state, 0)

                    # Update reward
                    cumulative_reward += test_reward

                print(f"Policy_reward for test: {cumulative_reward}")

                # Logging
                if not args.no_tensorboard:
                    writer.add_scalar('validation/policy_reward', cumulative_reward,
                                      global_steps)
                if log_filename:
                    with open(log_filename, "a") as f:
                        f.write(
                            f"{episode},{global_steps},{cumulative_reward},\n")

    if not args.no_tensorboard:
        writer.add_scalar('training/avg_episode_loss', cumulative_loss / steps,
                          episode)
    end = time.time()
    episode += 1

env.close()
test_env.close()

if args.model_path:
    torch.save(agent.online,
               append_timestamp(f"{args.model_path}/{run_tag}") + ".pth")
