import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.env_dynamics_wm import EnvDynamics
from agents.agents import Agent

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Adversarial attacks on the CarRacing-v0 environment')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--attack_type', type=str, default='general', metavar='N', help='type of the attack')
parser.add_argument('--adv_bound', type=float, default=0.1, metavar='N', help='epsilon value for perturbation limits')
# only use if attack_type is not general
parser.add_argument('--patch_type', type=str, default='box', metavar='N', help='type of patch in patch attack type')
parser.add_argument('--patch_size', type=int, default=24, metavar='N', help='size of patch in patch attack type')
parser.add_argument('--lmd', type=float, default=1.0, metavar='N', help='lmd for dynamics loss')
parser.add_argument('--unroll_length', type=int, default=10, metavar='N', help='Unroll length (T) for dynamics model')
parser.add_argument('--attack_length', type=int, default=7, metavar='N', help='attack length (N) when perturb persist')
parser.add_argument('--target_state', type=str, default='target_2', metavar='N',
                    help='select on target state in target_samples directory')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs the model needs to be trained')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for PGD attack')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# variables for patch attack, Need to move somewhere else later
box_dim = (48, 48)
box_position = (10, 10)
circle_centre = (24, 72)
circle_radius = 20

adv_dir = 'adv_attacks/perturbations/'


def test_attack():
    agent = Agent(args.img_stack, device)
    agent.load_param()
    env = EnvDynamics(args.seed, args.img_stack, args.unroll_length, device, args.lmd)

    # load adv input, by default general attack perturbation
    file = f'{adv_dir}{args.attack_type}'
    if args.attack_type == 'patch':
        file += f'_{args.patch_type}'
    file += f'_{args.adv_bound}.npy'
    delta_s = np.load(file)

    # initialize s_0 and draw corresponding a_0, waste few frames at the beginning if needed
    state = env.reset()
    for i in range(4):
        action = agent.select_action(state)
        state_, _, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        state = state_

    for t in range(1, 1000):
        d_s_i = np.zeros(state.shape)
        if t <= args.attack_length:
            # test the attack
            s_i = state
            d_s_i = delta_s[t - 1]
            s_with_d_s = s_i + d_s_i
            action = agent.select_action(s_with_d_s)
        else:
            action = agent.select_action(state)

        # show state and delta_s for unrolled length
        plt.imshow((state + d_s_i)[0], cmap='gray')
        plt.title('frame 1 of s + delta_s at T=' + str(t))
        plt.show()

        state_, _, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]), True)
        if args.render:
            env.render()
        state = state_
        if done:
            break


if __name__ == "__main__":
    test_attack()
