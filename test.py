import argparse
import numpy as np
import torch

from env.env_adv import Env
from agents.agent import Agent
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

if __name__ == "__main__":
    agent = Agent(args.img_stack, device)
    agent.load_param(device)
    env = Env(args.seed, args.img_stack, args.action_repeat)

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(10):
        score = 0
        state = env.reset()

        for t in range(1000):
            if t == 7:
                print('Pause the game')
                print(len(env.env.track))
                # time.sleep(10)
            action = agent.select_action(state, device)
            state_, reward, done, die, adv = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if t == 17:
                plt.imshow(128 * state_[0] + 1, cmap='gray')
                plt.title('state (first frame of 4)')
                plt.show()
                plt.imshow(adv[0], cmap='gray')
                plt.title('Physical perturbation that needs to be optimized (first frame of 4)')
                plt.show()

            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
