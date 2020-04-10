import torch
import torch.optim as optim
from torchvision.utils import make_grid
import numpy as np
import argparse
import warnings
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
parser.add_argument('--lmd', type=float, default=1.0, metavar='N', help='lmd for dynamics loss')
parser.add_argument('--unroll_length', type=int, default=20, metavar='N', help='Unroll length (T) for dynamics model')
parser.add_argument('--target_state', type=str, default='target_2', metavar='N',
                    help='select on target state in target_samples directory')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs the model needs to be trained')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for PGD attack')
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

target_dir = 'adv_attacks/target_samples/'
adv_dir = 'adv_attacks/perturbations/'


class AdvAttackDynamics:
    def __init__(self, attack_type, unroll_length, target_state):
        self.attack_type = attack_type
        self.unroll_length = unroll_length
        self.buffer_type = np.dtype(
            [('s', np.float64, (args.img_stack, 96, 96)), ('d_s', np.float64, (args.img_stack, 96, 96))])
        self.buffer = np.empty(self.unroll_length, dtype=self.buffer_type)
        self.buffer_counter = 0
        self.target_state = np.load(target_dir + target_state + '.npy')

    def load_networks(self):
        self.net = A2CNet(args.img_stack).float().to(device)
        if device == torch.device('cpu'):
            self.net.load_state_dict(torch.load('param/ppo_net_params.pkl', map_location='cpu'))
        else:
            self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))

    def update_buffer(self, state, delta_s, patch_type):
        # add state and delta_s to buffer
        if patch_type == 'box':
            # for now, let's predefine box size and shape
            temp = delta_s
            delta_s = np.zeros_like(temp)
            delta_s[:, box_position[0]: box_position[0] + box_dim[0],
            box_position[1]: box_position[1] + box_dim[1]] = temp[:, box_position[0]: box_position[0] + box_dim[0],
                                                             box_position[1]: box_position[1] + box_dim[1]]
        elif patch_type == 'circle':
            for i in range(delta_s.shape[1]):
                for j in range(delta_s.shape[2]):
                    if (circle_centre[0] - i) ** 2 + (circle_centre[1] - j) ** 2 >= circle_radius ** 2:
                        delta_s[:, i, j] = 0
        self.buffer[self.buffer_counter] = (state, delta_s)
        self.buffer_counter += 1

    def train(self):
        # optimize perturbation
        # get states and delta_s from the buffer
        s = torch.tensor(self.buffer['s'], dtype=torch.float).to(device)
        # get d_s (perturbation from buffer), d_s shape (T, 4, 96, 96)
        d_s = torch.tensor(self.buffer['d_s'], dtype=torch.float).to(device)
        # set grad true so that it can be differentiable
        d_s.requires_grad = True
        # set target state
        s_t = torch.cat(self.unroll_length * [torch.tensor(self.target_state).unsqueeze(0)])
        # print(s, d_s, s_t, sep='\n')

        for x in range(args.epochs):
            mse_loss = ((s - s_t) ** 2).sum()
            mse_loss.requires_grad = True
            # set up adam optimizer with d_s (perturbation) as parameters
            mse_optim = optim.Adam([d_s], lr=0.01)
            # set gradients zero before back propagation
            mse_optim.zero_grad()
            # perform back propagation
            mse_loss.backward()
            mse_optim.step()

            # only allow unmasked for the patch in case of patch attack
            with torch.no_grad():
                if self.attack_type == 'patch':
                    if args.patch_type == 'box':
                        temp = d_s
                        d_s = torch.zeros_like(temp)
                        d_s[:, :, box_position[0]: box_position[0] + box_dim[0],
                        box_position[1]: box_position[1] + box_dim[1]] = temp[:, :,
                                                                         box_position[0]: box_position[0] + box_dim[0],
                                                                         box_position[1]: box_position[1] + box_dim[1]]
                    elif args.patch_type == 'circle':
                        for i in range(d_s.shape[2]):
                            for j in range(d_s.shape[3]):
                                if (circle_centre[0] - i) ** 2 + (circle_centre[1] - j) ** 2 >= circle_radius ** 2:
                                    d_s[:, :, i, j] = 0
                # set constraints on perturbations
                d_s = torch.clamp(d_s, -args.adv_bound, args.adv_bound)

            mse_loss_scalar = mse_loss.detach().numpy().item() / self.unroll_length
            print(f'Loss: {mse_loss_scalar}')

        # save delta_s parameters
        with torch.no_grad():
            d_s_np = d_s.cpu().squeeze(0).numpy()
        print(d_s_np.shape)
        file_path = f'{adv_dir}{self.attack_type}_{args.adv_bound}.npy'
        np.save(file_path, d_s_np)
        print('delta_s saved in ', file_path)


def run_agent():
    agent = Agent(args.img_stack, device)
    agent.load_param()
    env = EnvDynamics(args.seed, args.img_stack, args.unroll_length, device, args.lmd)

    state = env.reset()

    # Prepare attack
    attack = AdvAttackDynamics(args.attack_type, args.unroll_length, args.target_state)
    attack.load_networks()

    for t in range(1000):
        # show states
        # plt.imshow(state[0], cmap='gray')
        # plt.title('frame 1 of state at T=' + str(t))
        # plt.show()

        # Initialize random perturbation for optimising the attack
        delta_s = np.random.random(state.shape) * 0.1
        # update buffer for training the attack
        attack.update_buffer(state, delta_s, args.patch_type)
        # get actions and value functions from NN based  on s + d_s.
        s_with_d_s = state + delta_s
        # observation limits
        s_with_d_s = np.clip(s_with_d_s, -1, 0.9921875)

        action = agent.select_action(s_with_d_s)
        state_, _, done, _ = env.step(state, action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        state = state_
        if done:
            break

    # train attack
    print('Training started ...')
    attack.train()


if __name__ == "__main__":
    run_agent()
