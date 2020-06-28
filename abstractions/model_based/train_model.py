import time
import os

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from ..common.replay_buffer import ReplayBuffer
from ..common.utils import model_based_parser, append_timestamp
from .model import ModelNet

def get_action(env):
    ''' random policy '''
    return env.action_space.sample()
    

if __name__ == "__main__":  

    args = model_based_parser.parse_args()

    if not args.no_tensorboard:
        writer = SummaryWriter(comment=args.run_tag)
    else:
        writer = None

    # Check if GPU can be used and was asked for
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    env = gym.make(args.env)
    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    
    model = ModelNet(args, device, env.observation_space, env.action_space)
    model = model.to(device)

    global_steps = 0
    episode = 0
    start = time.time()
    end = time.time() + 1

    while global_steps < args.max_steps:
        start = time.time()

        state = env.reset()
        done = False

        cumulative_loss = 0
        steps = 1
        while not done:
            steps += 1
            action = get_action(env)
            next_state, reward, done, _ = env.step(action)
            steps += 1
            replay_buffer.append(state, action, reward, next_state, int(done))

            global_steps += 1

            if len(replay_buffer) < args.batchsize or global_steps < args.warmup_period:
                continue

            minibatch = replay_buffer.sample(args.batchsize) 
            loss = model.train_batch(minibatch, global_steps, writer, args.gradient_clip)
            cumulative_loss += loss

        end = time.time()

        if writer and global_steps >= args.warmup_period and len(replay_buffer) >= args.batchsize:
            writer.add_scalar('episodewise/episode_loss', cumulative_loss, episode)

        info_str = "episode: {}, ".format(episode)
        info_str += "steps: {}, ".format(global_steps)
        info_str += "loss: {}, ".format(cumulative_loss)
        info_str += "FPS: {}".format(steps/(end - start))
        print(info_str)

        episode += 1

    if args.model_path:
        torch.save(model, append_timestamp(os.path.join(args.model_path, args.run_tag)) + ".pth")
