import argparse
import numpy as np
import torch
import os
import sys
from os.path import join
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.agents import Agent, RandomAgent
from env.env_adv import EnvRandom, Env

parser = argparse.ArgumentParser('Generate data to train VAE and dynamics model')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--sample-size', type=int, default=1e4, help='Total number of samples to generate')
parser.add_argument('--thread-no', type=int, default=0, help='Thread number to create directory based on')
parser.add_argument('--root-dir', type=str, required=True, help='Directory to store data')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--policy', type=str, required=True, choices=['random', 'pretrained'], help='Policy to be chosen for agent')
parser.add_argument('--noise-type', type=str, choices=['white', 'brown'], default='brown',
                    help='Noise type used for action sampling')
args = parser.parse_args()

np.random.seed(args.seed)
# GPU parameters
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# create respective directories and set dataset path
final_dir = args.root_dir
os.makedirs(final_dir, exist_ok=True)


def generate_data(agent, env):
    # initialize empty buffer of size sample size
    trajectory_type = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)),
                                ('s_', np.float64, (args.img_stack, 96, 96))])
    buffer = np.empty(int(args.sample_size), dtype=trajectory_type)

    # Loop through number of episodes
    for i in range(int(args.sample_size // 1000)):
        state = env.reset()
        # generate actions if it is random policy
        if args.policy == 'random':
            agent.generate_actions(args.noise_type)
        action, state_, done = None, None, None
        # Loop through each frame
        for t in range(1000):
            # select action based on policy
            if args.policy == 'random':
                action = agent.select_action(state)
                state_, _, done, _ = env.step(action)
            elif args.policy == 'pretrained':
                action = agent.select_action(state)
                state_, _, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            # render env if needed
            if args.render:
                env.render()
            # add (state, action, state_) to buffer
            buffer[i * 1000 + t] = (state, action, state_)
            # set next state as current state
            state = state_
            if t == 999 or done:
                break

    np.savez_compressed(join(final_dir, f'{args.policy}-{int(args.sample_size)}-ns-seed-{args.seed}-trajectories'), buffer)
    print(f'Data generated from {args.policy} policy of size {int(args.sample_size)} with seed {args.seed}')


if __name__ == '__main__':
    # Initialize environment for random and pretrained policy
    # For random policy, set action_repeat = 1 to have 1000 frames per episode
    env_random = EnvRandom(args.seed, args.img_stack, 1)
    # TO DO: environment set up for pretrained policy and separate logic for collecting frames
    env = Env(args.seed, args.img_stack, args.action_repeat)
    # Initialize agent
    agent = None
    if args.policy == 'pretrained':
        # NOTE: Do not use this. Still has implementation left to do
        agent = Agent(args.img_stack, device)
        agent.load_param()
        # Generate data
        generate_data(agent, env)
    elif args.policy == 'random':
        agent = RandomAgent(env, args.seed, device)
        # Generate data
        generate_data(agent, env_random)
