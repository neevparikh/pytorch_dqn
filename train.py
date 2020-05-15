import gym
import time
import numpy as np
import random
import torch
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.core.lightning import LightningModule

from utils import parse_args, make_ari, make_atari, append_timestamp
from model import DQN_CNN_model, DQN_MLP_model
from replay_buffer import RBDataset


class DQN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Setting cuda seeds
        if torch.cuda.is_available():
            torch.backends.cuda.deterministic = True
            torch.backends.cuda.benchmark = False

        # Get uuid for run
        if self.args.uuid == 'env':
            self.uuid_tag = self.hparams.env
        elif self.args.uuid == 'random':
            import uuid
            self.uuid_tag = uuid.uuid4()
        else:
            self.uuid_tag = self.hparams.uuid

        # Set tag for this run
        self.run_tag = self.hparams.env
        self.run_tag += '_' + self.args.uuid if self.args.uuid != 'env' else ''
        self.run_tag += '_ari' if self.args.ari else ''
        self.run_tag += '_seed_' + str(self.args.seed)

        # Setting random seed
        torch.manual_seed(self.hparams.seed)
        random.seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)

        # Save path
        if self.hparams.model_path:
            os.makedirs(self.hparams.model_path, exist_ok=True)

        if self.hparams.output_path:
            os.makedirs(self.hparams.output_path, exist_ok=True)

        state_space = env.observation_space
        action_space = env.action_space
        num_actions = env.action_space.n
        if self.hparams.model_type == "mlp":
            self.online = DQN_MLP_model(state_space, action_space,
                                        num_actions)
            self.target = DQN_MLP_model(state_space, action_space,
                                        num_actions)
        elif self.hparams.model_type == "cnn":
            assert num_frames
            self.num_frames = num_frames
            self.online = DQN_CNN_model(state_space,
                                        action_space,
                                        num_actions,
                                        num_frames=num_frames)
            self.target = DQN_CNN_model(state_space,
                                        action_space,
                                        num_actions,
                                        num_frames=num_frames)
        else:
            raise NotImplementedError(self.hparams.model_type)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Model related
        parser.add_argument('--model-type',
                            help="Type of architecture",
                            type=str,
                            default='mlp',
                            choices=["cnn", "mlp"],
                            required=True)
        parser.add_argument('--vanilla-DQN',
                            help='Use the vanilla dqn, instead of double DQN',
                            action='store_true',
                            required=False)

        # Network related
        parser.add_argument('--batchsize',
                            help='Number of experiences from replay buffer',
                            type=int,
                            default=256,
                            required=False)
        parser.add_argument('--lr',
                            help='Learning rate for the optimizer',
                            type=float,
                            default=5e-4,
                            required=False)
        parser.add_argument('--target-moving-average',
                            help='EMA parameter for target network',
                            type=float,
                            default=5e-3,
                            required=False)

        # Training and model related
        parser.add_argument('--test-policy-steps',
                            help='Policy is tested every these many steps',
                            type=lambda x: int(float(x)),
                            default=1000,
                            required=False)
        parser.add_argument('--warmup-period',
                            help='Number of steps to act randomly, not train',
                            type=lambda x: int(float(x)),
                            default=50000,
                            required=False)
        parser.add_argument('--gradient-clip',
                            help='How much to clip the gradients by, 0 is none',
                            type=float,
                            default=0,
                            required=False)
        parser.add_argument('--reward-clip',
                            help='How much to clip reward, clipped in \
                            [-rc, rc], 0 results in unclipped',
                            type=float,
                            default=0,
                            required=False)
        parser.add_argument('--epsilon-decay',
                            help='Parameter for epsilon decay',
                            type=lambda x: int(float(x)),
                            default=1e3,
                            required=False)
        parser.add_argument('--epsilon-decay-end',
                            help='Parameter for epsilon decay end',
                            type=float,
                            default=0.1,
                            required=False)
        parser.add_argument('--replay-buffer-size',
                            help='Max size of replay buffer',
                            type=lambda x: int(float(x)),
                            default=500000,
                            required=False)
        return parser

    def train_dataloader(self):
        # Initialize environment
        if type(self.args.env) == str:
            self.env = gym.make(self.hparams.env)
            self.test_env = gym.make(self.hparams.env)
        else:
            self.env = self.hparams.env
            self.test_env = self.hparams.env

        self.env.seed(self.hparams.seed)
        self.test_env.seed(self.hparams.seed)

        if self.hparams.model_type == 'cnn':
            assert self.hparams.num_frames
            if not self.hparams.no_atari:
                print("Using atari preprocessing")
                env = make_atari(env, self.hparams.num_frames)
                test_env = make_atari(test_env, self.hparams.num_frames)

        if self.hparams.ari:
            print("Using ARI")
            env = make_ari(env)
            test_env = make_ari(test_env)

        if type(env.action_space) != gym.spaces.Discrete:
            raise NotImplementedError("DQN for continuous action_spaces hasn't\
                    been implemented")
        
        return RBDataset(self.args.replay_buffer_size, self.args.batchsize)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.online.parameters(),
                lr=self.hparams.lr)
        return [optimizer]




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

                test_action = agent.online.act(state, 0)
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
                    test_action = agent.online.act(state, 0)

                    # Update reward
                    cumulative_reward += test_reward

                print(f"Policy_reward for test: {cumulative_reward}")

                # Logging
                writer.add_scalar('validation/policy_reward', cumulative_reward,
                                  global_steps)
                if log_filename:
                    with open(log_filename, "a") as f:
                        f.write(
                            f"{episode},{global_steps},{cumulative_reward},\n")

    writer.add_scalar('training/avg_episode_loss', cumulative_loss / steps,
                      episode)
    end = time.time()
    episode += 1

env.close()
test_env.close()

if args.model_path:
    torch.save(agent.online,
               append_timestamp(f"{args.model_path}/{run_tag}") + ".pth")
