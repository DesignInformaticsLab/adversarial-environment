import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.env_adv import Env
from agents.agents import Agent

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--mode', type=str, default='collect', metavar='N', help='either collect or view')
parser.add_argument('--ns', type=int, default=5, metavar='N', help='number of samples needed')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


target_dir = 'adv_attacks/target_samples/'


def collect():
    agent = Agent(args.img_stack, device)
    agent.load_param()
    env = Env(args.seed, args.img_stack, args.action_repeat)

    counter = 0
    for i_ep in range(args.ns):
        score = 0
        state = env.reset()
        rand_action_start_t = int(np.random.uniform(0, 100, 1).view())
        print(rand_action_start_t)
        for t in range(1000):
            # draw a random number between 0 and 100 to start off a random action and then collect sample when die
            if t >= rand_action_start_t:
                action = np.random.uniform(0.5, 1, (3,))
            else:
                action = agent.select_action_np(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                counter += 1
                np.save(f'{target_dir}target_{counter}.npy', state_)
            if done or die:
                break

        if counter >= args.ns:
            print('Target samples collected')
            break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))


def view():
    for filename in sorted(os.listdir(target_dir), key=lambda x: int(x.split('_')[1].split('.')[0])):
        target_state = np.load(target_dir + filename)
        plt.imshow(target_state[0], cmap='gray')
        plt.title(filename)
        plt.show()


if __name__ == "__main__":
    if args.mode == 'collect':
        collect()
    elif args.mode == 'view':
        view()
    else:
        print('Invalid mode')
