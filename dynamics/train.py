import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import argparse
import warnings
import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.env_adv import Env
from agents.agents import Agent, RandomAgent
from dynamics.dynamics_model import DynamicsModel

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Learn Dynamics Model of Environment')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs the model needs to be trained')
parser.add_argument('--sample_size', type=float, default=1e4, help='sample size for collect trajectories')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the model')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for training the model')
parser.add_argument('--lmd', type=float, default=1.0, help='specifies weight balance between AE loss and dynamics loss')
args = parser.parse_args()

# GPU parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Hyper Parameters
epochs = args.epochs
sample_size = args.sample_size
batch_size = args.batch_size
lr = args.lr
lmd = args.lmd

weights_file_path = 'dynamics/param/learn_dynamics_lmd_{}.pkl'.format(lmd)
data_file_path = 'dynamics/trajectories/{}-ns-trajectories-seed-{}.npy'.format(int(sample_size), args.seed)


def collect_trajectories(agent, env, ns):
    counter = 0
    # initialize empty buffer of size ns
    trajectory_type = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)),
                                ('s_', np.float64, (args.img_stack, 96, 96))])
    buffer = np.empty(int(ns), dtype=trajectory_type)
    state = env.reset()

    while counter < ns:
        action = agent.select_action(state)
        state_, _, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        buffer[counter] = (state, action, state_)
        state = state_
        if done or die:
            state = env.reset()
        counter += 1

    return buffer


def save_param(net):
    file_path = weights_file_path
    torch.save(net.state_dict(), file_path)
    print('NN saved in ', file_path)


def load_param(net):
    if device == torch.device('cpu'):
        net.load_state_dict(torch.load(weights_file_path, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(weights_file_path))


def combined_loss(pred_state, gt_state, pred_state_, gt_state_):
    l2_loss = nn.MSELoss()

    AE_loss = l2_loss(pred_state, gt_state)
    dynamics_loss = l2_loss(pred_state_, gt_state_)

    return AE_loss + lmd * dynamics_loss


def train():
    # Initialize pre-trained policy agent and random agent
    agent = Agent(args.img_stack, device)
    agent.load_param()
    rand_agent = RandomAgent(args.img_stack, device)
    # Initialize environment
    env = Env(args.seed, args.img_stack, args.action_repeat)
    # Initialize dynamics model
    dynamics = DynamicsModel(args.img_stack).to(device)
    # Initialize optimizer
    optimizer = optim.Adam(dynamics.parameters(), lr=lr)

    # Look for trajectory file if present
    if os.path.isfile(data_file_path):
        print('Trajectory found, Loading data ...')
        trajectory = np.load(data_file_path)
        print('Trajectories loaded')
    else:
        print('Collecting Pre-trained Policy Trajectories')
        buffer = collect_trajectories(agent, env, sample_size)
        print('Collecting Random Trajectories')
        rand_buffer = collect_trajectories(rand_agent, env, sample_size)

        trajectory = np.concatenate([buffer, rand_buffer])
        np.save(data_file_path, trajectory)
        print('Trajectories Data saved in ', data_file_path)

    s = torch.tensor(trajectory['s'], dtype=torch.float).to(device)
    a = torch.tensor(trajectory['a'], dtype=torch.float).to(device)
    next_s = torch.tensor(trajectory['s_'], dtype=torch.float).to(device)

    dynamics.train()
    if os.path.isfile(weights_file_path):
        load_param(dynamics)

    print('Training')
    for i in tqdm.trange(epochs):
        running_loss, no_of_batches = 0, 0
        for index in BatchSampler(SubsetRandomSampler(range(int(sample_size))), batch_size, False):
            p_s, p_s_ = dynamics(s[index], a[index])
            pred_s = torch.cat((s[index][:, :3, :, :], p_s), dim=1)
            pred_s_ = torch.cat((next_s[index][:, :3, :, :], p_s_), dim=1)

            loss = combined_loss(pred_s, s[index], pred_s_, next_s[index])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # enforce constraints on big_lambda
            mask = torch.eye(16).to(device)
            with torch.no_grad():
                dynamics.big_lambda.weight.data = dynamics.big_lambda.weight.data * mask
                torch.clamp(dynamics.big_lambda.weight.data, 0, 1)

            running_loss += loss.item()
            no_of_batches += 1

        running_loss = running_loss / no_of_batches
        print(' loss: {}'.format(running_loss))

        if running_loss < 1e-4 or i == epochs - 1:
            save_param(dynamics)
            break

        # if i % 100 == 0:
        #     num = np.random.randint(0, batch_size)
        #     bounds = np.random.randint(0, ns - batch_size)
        #     with torch.no_grad():
        #         plt.title('Predicted')
        #         _, pred = dl(torch.from_numpy(trajectory[bounds:bounds + batch_size]['s']).float().cuda(),
        #                      torch.from_numpy(trajectory[bounds:bounds + batch_size]['a']).float().cuda())
        #         plt.imshow(pred.cpu()[num])
        #         plt.show()
        #     plt.title('Groundtruth')
        #     plt.imshow(trajectory[bounds:bounds + batch_size]['s_'][num, 3, :, :])
        #     plt.show()
    print('Training done')


if __name__ == '__main__':
    train()
