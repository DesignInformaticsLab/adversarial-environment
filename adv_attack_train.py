import torch
import torch.optim as optim
from torchvision.utils import make_grid
import numpy as np
import argparse
import warnings

from env.env import Env
from networks.actor_critic import A2CNet
from agents.agent import Agent
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Adversarial attacks on the CarRacing-v0 environment')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--attack_type', type=str, default='general', metavar='N', help='type of the attack')
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


class AdvAttack:
    def __init__(self, attack_type):
        self.attack_type = attack_type
        # create a buffer to store state and delta_ts
        self.buffer_capacity = 128
        self.buffer_type = np.dtype(
            [('s', np.float64, (args.img_stack, 96, 96)), ('d_s', np.float64, (args.img_stack, 96, 96))])
        self.buffer = np.empty(self.buffer_capacity, dtype=self.buffer_type)
        self.buffer_counter, self.is_buffer_full = 0, False
        # take a left turn as target action [0.25, 0.5, 0.25]. This can be changed
        self.target_action = torch.from_numpy(np.array([0.25, 0.5, 0.25], dtype=np.double))
        # counter to track loss in tensorboard
        self.tensorboard_counter = 0

    def initialize_perturbation(self, shape, ):
        self.delta_s = np.random.random(shape) * 0.1
        # will deal later for multiple attacks
        if self.attack_type == 'patch':
            self.modify_perturbation(args.patch_type)

    def modify_perturbation(self, patch_type):
        if patch_type == 'box':
            # for now, let's predefine box size and shape
            temp = self.delta_s
            self.delta_s = np.zeros_like(temp)
            self.delta_s[:, box_position[0]: box_position[0] + box_dim[0],
            box_position[1]: box_position[1] + box_dim[1]] = temp[:, box_position[0]: box_position[0] + box_dim[0],
                                                             box_position[1]: box_position[1] + box_dim[1]]
        elif patch_type == 'circle':
            for i in range(self.delta_s.shape[1]):
                for j in range(self.delta_s.shape[2]):
                    if (circle_centre[0] - i) ** 2 + (circle_centre[1] - j) ** 2 >= circle_radius ** 2:
                        self.delta_s[:, i, j] = 0

    def load_networks(self):
        self.net = A2CNet(args.img_stack).double().to(device)
        if device == torch.device('cpu'):
            self.net.load_state_dict(torch.load('param/ppo_net_params.pkl', map_location='cpu'))
        else:
            self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))

    def update_buffer(self, state):
        # add state and delta_s to buffer
        self.buffer[self.buffer_counter] = (state, self.delta_s)
        self.buffer_counter += 1
        # check if the buffer is filled
        if self.buffer_counter == self.buffer_capacity:
            self.buffer_counter = 0
            self.is_buffer_full = True

    def train(self):
        if self.is_buffer_full:
            # optimize perturbation
            # get states from the buffer
            s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
            # iterate through 10 epochs
            for x in range(10):
                # get d_s (perturbation from buffer), d_s shape ~ (128, 4, 96, 96)
                d_s = torch.tensor(self.buffer['d_s'], dtype=torch.double).to(device)
                # set grad true so that it can be differentiable
                d_s.requires_grad = True

                # get actions and value functions from NN based  on s + d_s.
                (alpha, beta), v_adv = self.net(s + d_s)
                a_adv = alpha / (alpha + beta)

                d_s, mse_loss = self.optimize_perturbation(a_adv, d_s)

                # save delta_s as average of d_s in the entire batch
                self.delta_s = np.average(d_s.detach().numpy(), axis=0)

                # print delta_s and mse loss
                mse_loss_scalar = mse_loss.detach().numpy().item() / self.buffer_capacity
                print('Sum of delta_s:', np.sum(self.delta_s), ', MSE Loss: ', mse_loss_scalar)
                # check if mse loss is less than 10e-3, then save delta_s param
                if mse_loss_scalar < 0.01:
                    file_path = 'param/adv_' + self.attack_type
                    if self.attack_type == 'patch':
                        file_path += '_' + args.patch_type
                    file_path += '.npy'
                    np.save(file_path, self.delta_s)
                    print('delta_s saved in ', file_path)
                    exit(0)

                # update buffer with new delta_s
                for i in range(self.buffer_capacity):
                    self.buffer[i]['d_s'] = self.delta_s

                self.tensorboard_counter += 1
            if self.tensorboard_counter % 10 == 0:
                writer.add_scalar('mse loss', mse_loss_scalar, self.tensorboard_counter)
                print('Loss added to tensorboard')

    def optimize_perturbation(self, adv_action, perturb):
        # mean square loss
        mse_loss = ((adv_action - self.target_action) ** 2).sum()
        # set up adam optimizer with d_s (perturbation) as parameters
        mse_optim = optim.Adam([perturb], lr=0.01)
        # set gradients zero before back propagation
        mse_optim.zero_grad()
        # perform back propagation
        mse_loss.backward()
        # only allow gradients for the patch in case of patch attack
        if self.attack_type == 'patch':
            if args.patch_type == 'box':
                temp = perturb.grad
                perturb.grad = torch.zeros_like(temp)
                perturb.grad[:, :, box_position[0]: box_position[0] + box_dim[0],
                box_position[1]: box_position[1] + box_dim[1]] = temp[:, :,
                                                                 box_position[0]: box_position[0] + box_dim[0],
                                                                 box_position[1]: box_position[1] + box_dim[1]]
            elif args.patch_type == 'circle':
                for i in range(perturb.shape[2]):
                    for j in range(perturb.shape[3]):
                        if (circle_centre[0] - i) ** 2 + (circle_centre[1] - j) ** 2 >= circle_radius ** 2:
                            perturb.grad[:, :, i, j] = 0

        mse_optim.step()
        return perturb, mse_loss


def run_agent():
    agent = Agent(args.img_stack, device)
    agent.load_param(device)
    env = Env(args.seed, args.img_stack, args.action_repeat)

    state = env.reset()

    # Prepare attack
    attack = AdvAttack(args.attack_type)
    attack.initialize_perturbation(state.shape)
    attack.load_networks()

    for i_ep in range(50):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state, device)
            # update buffer for training the attack
            attack.update_buffer(state)

            # write to tensorboard
            input_imgs_to_net = torch.tensor((attack.buffer['s'] + attack.buffer['d_s']))
            input_imgs_grid = make_grid(input_imgs_to_net[0].reshape(4, 1, 96, 96))
            writer.add_image('Four stack of input state with adversarial', input_imgs_grid)
            writer.add_graph(attack.net, input_imgs_to_net)
            writer.close()

            # train attack
            attack.train()

            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))


if __name__ == "__main__":
    run_agent()
