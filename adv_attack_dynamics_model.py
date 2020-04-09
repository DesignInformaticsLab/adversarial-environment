import torch
import torch.optim as optim
from torchvision.utils import make_grid
import numpy as np
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from env.env_dynamics import EnvDynamics
from networks.actor_critic import A2CNet
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
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# variables for patch attack, Need to move somewhere else later
box_dim = (48, 48)
box_position = (10, 10)
circle_centre = (24, 72)
circle_radius = 20

# tensorboard variables
writer_name = 'runs/adv_' + args.attack_type
if args.attack_type == 'patch':
    writer_name += '_' + args.patch_type
writer = SummaryWriter(writer_name)


def run_agent():
    agent = Agent(args.img_stack, device)
    agent.load_param()
    env = EnvDynamics(args.seed, args.img_stack, device, 1.0)

    for i_ep in range(10):
        state = env.reset()

        for t in range(1000):
            plt.imshow(state[0], cmap='gray')
            plt.title('frame 1 of state at T=' + str(t))
            plt.show()
            action = agent.select_action(state)
            state_, _, done, _ = env.step(state, action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            state = state_
            if done:
                break


if __name__ == "__main__":
    run_agent()
