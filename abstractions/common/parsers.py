import argparse

float_to_int = lambda x: int(float(x))

# yapf: disable
common_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
common_parser.add_argument('--env', type=str, required=True,
        help='The gym environment to train on')
common_parser.add_argument('--gamma', type=float, required=False, default=0.99,
        help='Gamma parameter')
common_parser.add_argument('--no-tensorboard', action='store_true', required=False,
        help='Do not use Tensorboard')
common_parser.add_argument('--gpu', action='store_true', required=False,
        help='Use the gpu or not')
common_parser.add_argument('--max-steps', type=float_to_int, required=False, default=40000,
        help='Number of steps to run for')
common_parser.add_argument('--model-path', type=str, required=False,
        help='The path to the save the pytorch model')
common_parser.add_argument('--output-path', default='logs', type=str, required=False,
        help='The output directory to store training stats')
common_parser.add_argument('--seed', type=int, required=False, default=10,
        help='The random seed for this run')
common_parser.add_argument('--run-tag', type=str, required=True,
        help="Run tag for the experient run.")
common_parser.add_argument('--render', action='store_true', required=False,
        help='Render visual or not')
common_parser.add_argument('--render-episodes', type=int, required=False, default=5,
        help='Render every these many episodes')
common_parser.add_argument('--replay-buffer-size', type=float_to_int, required=False, default=50000,
        help='Max size of replay buffer')
common_parser.add_argument('--lr', type=float, required=False, default=3e-5,
        help='Learning rate for the optimizer')
common_parser.add_argument('--batchsize', type=int, required=False, default=32,
        help='Number of experiences sampled from replay buffer')
common_parser.add_argument('--gradient-clip', type=float, required=False, default=0,
        help='How much to clip the gradients by, 0 is none')
common_parser.add_argument('--reward-clip', type=float, required=False, default=0,
        help='How much to clip reward, i.e. [-rc, rc]; 0 is unclipped')
common_parser.add_argument('--model-shape', type=str, default='medium',
        choices=['tiny', 'small', 'medium', 'large', 'giant'],
        help="Shape of architecture (mlp only)")
common_parser.add_argument('--num-frames', type=int, required=False, default=4,
        help='Number of frames to stack (CNN only)')
common_parser.add_argument('--warmup-period', type=float_to_int, required=False, default=2000,
        help='Number of steps to act randomly and not train')
common_parser.add_argument('--episodes-per-eval', type=int, default=10,
        help='Number of episodes per evaluation (i.e. during test)')

dqn_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
dqn_parser.add_argument('--ari', action='store_true', required=False,
        help='Whether to use annotated RAM')
dqn_parser.add_argument('--action-stack', action='store_true', required=False,
        help='Whether to stack action as previous plane')
dqn_parser.add_argument('--model-type', type=str, required=True, default='mlp',
        choices=['cnn', 'mlp'],
        help="Type of architecture")
dqn_parser.add_argument('--load-checkpoint-path', type=str, required=False,
        help='Path to checkpoint')
dqn_parser.add_argument('--no-atari', action='store_true', required=False,
        help='Do not use atari preprocessing')
dqn_parser.add_argument('--checkpoint-steps', type=float_to_int, required=False, default=20000,
        help='Checkpoint every so often')
dqn_parser.add_argument('--test-policy-steps', type=float_to_int, required=False, default=1000,
        help='Policy is tested every these many steps')
dqn_parser.add_argument('--epsilon-decay-length', type=float_to_int, required=False, default=5000,
        help='Number of steps to linearly decay epsilon')
dqn_parser.add_argument('--final-epsilon-value', type=float, required=False, default=0.05,
        help='Final epsilon value, between 0 and 1')
dqn_parser.add_argument('--target-moving-average', type=float, required=False, default=0.01,
        help='EMA parameter for target network')
dqn_parser.add_argument('--vanilla-DQN', action='store_true', required=False,
        help='Use the vanilla dqn update instead of double DQN')

sac_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
sac_parser.add_argument('--policy', default="Gaussian", choices=["Gaussian", "Deterministic"],
        help='Policy Type')
sac_parser.add_argument('--target-moving-average', type=float, required=False, default=0.005,
        help='EMA parameter for target network')
sac_parser.add_argument('--alpha', type=float, default=0.2,
        help='Temperature parameter α determines the relative importance of the entropy term \
                against the reward (default: 0.2)'                                                                                                    )
sac_parser.add_argument('--automatic-entropy-tuning', action='store_true',
        help='Automaically adjust α (default: False)')
sac_parser.add_argument('--hidden-size', type=int, default=256,
        help='hidden size (default: 256)')
sac_parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'cnn'],
        help="Type of architecture")
sac_parser.add_argument('--updates-per-step', type=int, default=1,
        help='model updates per simulator step (default: 1)')
sac_parser.add_argument('--target-update-interval', type=int, default=1,
        help='Value target update per no. of updates per step (default: 1)')
sac_parser.add_argument('--test-policy-steps', type=float_to_int, required=False, default=1000,
        help='Policy is tested every these many steps')
sac_parser.add_argument('--no-atari', action='store_true', required=False,
        help='Do not use atari preprocessing')
sac_parser.add_argument('--ari', action='store_true', required=False,
        help='Whether to use annotated RAM')
sac_parser.add_argument('--action-stack', action='store_true', required=False,
        help='Whether to stack action as previous plane')

ppo_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
ppo_parser.add_argument('--discrete', action='store_true', required=False,
        help='Discrete action space')
ppo_parser.add_argument('--gradient-updates', type=float_to_int, default=80,
        help='model updates per simulator step')
ppo_parser.add_argument('--update-frequency', type=float_to_int, default=4e3,
        help='number of steps between model updates')
ppo_parser.add_argument('--action-std', type=float, default=0.5,
        help='standard deviation for action (continuous only)')
ppo_parser.add_argument('--eps-clip', type=float, default=0.2,
        help='Eps clip parameter for PPO')
ppo_parser.add_argument('--hidden-size', type=int, default=256,
        help='hidden size (default: 256)')
ppo_parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp'],
        help="Type of architecture")
ppo_parser.add_argument('--no-atari', action='store_true', required=False,
        help='Do not use atari preprocessing')
ppo_parser.add_argument('--ari', action='store_true', required=False,
        help='Whether to use annotated RAM')
ppo_parser.add_argument('--action-stack', action='store_true', required=False,
        help='Whether to stack action as previous plane')
ppo_parser.add_argument('--test-policy-steps', type=float_to_int, required=False, default=1000,
        help='Policy is tested every these many steps')

model_based_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser], add_help=False)
# yapf: enable


