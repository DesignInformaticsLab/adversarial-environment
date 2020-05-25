import argparse
import numpy as np
import torch
import os
import sys
from os.path import join

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.agents import Agent, RandomAgent
from env.env_adv import EnvRandom, Env

parser = argparse.ArgumentParser('Generate data to train VAE and dynamics model')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--episodes', type=int, default=1000, help='Total number of episodes or rollouts')
parser.add_argument('--thread-no', type=int, default=0, help='Thread number to create directory based on')
parser.add_argument('--root-dir', type=str, help='Directory to store data')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--policy', type=str, choices=['random', 'pretrained'], help='Policy to be chosen for agent')
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
final_dir = join(args.root_dir, args.policy)
os.makedirs(final_dir, exist_ok=True)
if args.policy == 'random':
    final_dir = join(final_dir, args.noise_type)
    os.makedirs(final_dir, exist_ok=True)
final_dir = join(final_dir, f'thread_{args.thread_no}')
os.makedirs(final_dir, exist_ok=True)


def generate_data(agent, env):
    # Loop through number of episodes
    for i in range(args.episodes):
        state = env.reset()
        # generate actions if it is random policy
        if args.policy == 'random':
            agent.generate_actions(args.noise_type)
        # list to collect frames
        state_seq, action_seq, next_state_seq = [], [], []
        action, state_, done = None, None, None
        # Loop through each frame
        for t in range(1000):
            # select action based on policy
            if args.policy == 'random':
                action = agent.select_action()
                state_, _, done, _ = env.step(action)
            elif args.policy == 'pretrained':
                action = agent.select_action(state)
                state_, _, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            # render env if needed
            if args.render:
                env.render()
            # add state, action, state_) to respective list
            state_seq.append(state)
            action_seq.append(action)
            next_state_seq.append(state_)
            # set next state as current state
            state = state_
            if t == 999 or done:
                print('> End of thread {}, episode {}, {} frames...'.format(args.thread_no, i + 1, len(state_seq)))
                np.savez(join(final_dir, 'episode_{}'.format(i + 1)),
                         states=np.array(state_seq),
                         actions=np.array(action_seq),
                         next_states=np.array(next_state_seq))
                break


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
