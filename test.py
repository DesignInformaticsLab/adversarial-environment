import argparse

import numpy as np
import torch

from env.env_dynamics_wm import EnvDynamics
from agents.agents import Agent, RandomAgent
from agents.agents import Agent
from env.env_adv import Env

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
parser.add_argument('--mode', default='pretrained', type=str, choices=['pretrained', 'random'], help='choose mode for rendering')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

def pretrained():
    agent = Agent(args.img_stack, device)
    agent.load_param()
    env = Env(args.seed, args.img_stack, args.action_repeat)

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(10):
        score = 0
        state = env.reset()

        for t in range(1000):
            #action = agent.select_action(state)
            action = torch.tensor([0.4, 0, 0], dtype=torch.float)
            state_, reward, done, die = env.step(action * torch.tensor([2., 1., 1.]) + torch.tensor([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))

def random():
    env = EnvDynamics(args.seed, args.img_stack, args.action_repeat, device)
    agent = RandomAgent(env, args.seed, device)
    agent.generate_actions()

    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in range(10):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state)
            state_, reward, done, die = env.step(action)
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))

if __name__ == "__main__":
    if args.mode == 'pretrained':
        pretrained()
    elif args.mode == 'random':
        random()

