import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model_dynamics_v2 import *

import numpy as np
import argparse
import warnings
import tqdm
import matplotlib.pyplot as plt
import os

from env.env_adv import Env
from agents.agent import Agent, Random_Agent

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Learn Dynamics of Environment')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--mode', type=str, default='train', help='sets nn mode (eval or train)')
parser.add_argument('--lmd', type=float, default=1, help='specifies weight balance between AE loss and dynamics loss')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

traj = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('s_', np.float64, (args.img_stack, 96, 96))])

epochs = 500
ns = 1e4
batch_size = 50
lmd = args.lmd
lr = 1e-2

weights_file_path = 'param/learn_dynamics_lmd_{}.pkl'.format(lmd)
data_file_path = './data/{}-ns-trajectories-seed-{}.npy'.format(int(ns), args.seed)

def collect_trajectories(agent, env, ns, device):
    counter = 0
    buffer = np.empty(int(ns/2), dtype=traj)
    state = env.reset()

    while counter < ns/2:
        action = agent.select_action(state, device)
        state_, _, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        buffer[counter] = (state, action, state_)
        state = state_
        if done or die:
            state = env.reset()
        counter += 1

    return buffer

def save_param(dl):
    file_path = weights_file_path
    torch.save(dl.state_dict(), file_path)
    print('NN saved in ', file_path)

def load_param(dl, device):
    if device == torch.device('cpu'):
        dl.load_state_dict(torch.load(weights_file_path, map_location='cpu'))
    else:
        dl.load_state_dict(torch.load(weights_file_path))

def loss_func(pred_state, gt_state, pred_state_, gt_state_):
    l2_loss = nn.MSELoss()

    AE_loss = l2_loss(pred_state, gt_state)
    dynamics_loss = l2_loss(pred_state_, gt_state_)

    return AE_loss + lmd*dynamics_loss

def train():
    agent = Agent(args.img_stack, device)
    agent.load_param(device)
    rand_agent = Random_Agent(args.img_stack, device)

    env = Env(args.seed, args.img_stack, args.action_repeat)

    dl = ModelDynamics(4, batch_size, device)

    if use_cuda:
        dl.cuda()

    optimizer = optim.Adam(dl.parameters(), lr=lr)

    if os.path.isfile(data_file_path):
        print('Trajectory found, Loading data...')
        trajectory = np.load(data_file_path)
    else:
        print('Collecting Trajectories')
        buffer = collect_trajectories(agent, env, ns, device)
        print('Collecting Random Trajectories')
        rand_buffer = collect_trajectories(rand_agent, env, ns, device)

        trajectory = np.concatenate([buffer, rand_buffer])
        np.save(data_file_path, trajectory)
        print('Data saved in ', data_file_path)

    s = torch.tensor(trajectory['s'], dtype=torch.float).to(device)
    a = torch.tensor(trajectory['a'], dtype=torch.float).to(device)
    groundtruth_s = torch.tensor(trajectory['s_'], dtype=torch.float).to(device)

    dl.train()
    if os.path.isfile(weights_file_path):
        load_param(dl, device)

    min_loss = 10000
    print('Training')
    for i in tqdm.trange(epochs):
        running_loss = 0
        for index in BatchSampler(SubsetRandomSampler(range(int(ns))), batch_size, False):
            pred_s, dyn_pred = dl(s[index].cuda(), a[index].cuda())

            pred_s = torch.cat((s[index][:, :3, :, :], pred_s.unsqueeze(1)), dim=1)
            pred_s_ = torch.cat((groundtruth_s[index][:, :3, :, :], dyn_pred.unsqueeze(1)), dim=1)

            loss = loss_func(pred_s, s[index], pred_s_, groundtruth_s[index])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(' loss: {}'.format(running_loss))

        if min_loss > running_loss:
            min_loss = running_loss
            save_param(dl)

        if i % 100 == 0:
            num = np.random.randint(0, batch_size)
            bounds = np.random.randint(0, ns-batch_size)
            with torch.no_grad():
                plt.title('Predicted')
                _, pred = dl(torch.from_numpy(trajectory[bounds:bounds + batch_size]['s']).float().cuda(),
                              torch.from_numpy(trajectory[bounds:bounds + batch_size]['a']).float().cuda())
                plt.imshow(pred.cpu()[num])
                plt.show()
            plt.title('Groundtruth')
            plt.imshow(trajectory[bounds:bounds + batch_size]['s_'][num, 3, :, :])
            plt.show()
    print('done')

def eval():
    device = torch.device('cpu')
    agent = Agent(args.img_stack, device)
    agent.load_param(device)
    rand_agent = Random_Agent(args.img_stack, device)

    env = Env(args.seed, args.img_stack, args.action_repeat)

    dl = ModelDynamics(4, batch_size)
    l2_loss = nn.MSELoss()

    if os.path.isfile(data_file_path):
        print('Trajectory found, Loading data...')
        trajectory = np.load(data_file_path)
    else:
        print('Collecting Trajectories')
        buffer = collect_trajectories(agent, env, ns, device)
        print('Collecting Random Trajectories')
        rand_buffer = collect_trajectories(rand_agent, env, ns, device)

        trajectory = np.concatenate([buffer, rand_buffer])
        np.save(data_file_path, trajectory)
        print('Data saved in ', data_file_path)

    s = torch.tensor(trajectory['s'], dtype=torch.float).to(device)
    a = torch.tensor(trajectory['a'], dtype=torch.float).to(device)
    groundtruth_s = torch.tensor(trajectory['s_'], dtype=torch.float).to(device)

    dl.eval()
    load_param(dl, device)

    total_loss = 0
    i = 0
    file = open("./imgs_v2/Losses.txt", "w")
    file.write("Seed {}".format(args.seed))
    for index in BatchSampler(SubsetRandomSampler(range(int(ns))), batch_size, False):
        with torch.no_grad():
            pred_s, dyn_pred = dl(s[index], a[index])

            pred_s = torch.cat((s[index][:, :3, :, :],pred_s.unsqueeze(1)), dim=1)
            pred_s_ = torch.cat((groundtruth_s[index][:, :3, :, :], dyn_pred.unsqueeze(1)), dim=1)

            loss = loss_func(pred_s, s[index], pred_s_, groundtruth_s[index])

        total_loss += loss.item()/ns

        print(' loss: {}'.format(loss.item()/ns))
        file.write("\nLoss for Batch {}: {}".format(i, loss.item()/ns))

        num = np.random.randint(0, batch_size)
        bounds = np.random.randint(0, ns - batch_size)
        with torch.no_grad():
            plt.title('Predicted')
            _, pred = dl(torch.from_numpy(trajectory[bounds:bounds + batch_size]['s']).float(),
                               torch.from_numpy(trajectory[bounds:bounds + batch_size]['a']).float())
            plt.imshow(pred[num], cmap='gray')
            plt.savefig('./imgs_v2/Pred from Batch {}'.format(i))
        plt.title('Groundtruth')
        plt.imshow(trajectory[bounds:bounds + batch_size]['s_'][num, 3, :, :], cmap='gray')
        plt.savefig('./imgs_v2/GT from Batch {}'.format(i))

        i += 1

    print('Total Loss: {}'.format(total_loss))
    file.write("\nTotal Loss: {}".format(total_loss))
    file.close()

if __name__ == '__main__':
    if args.mode == 'train':
        print('Train Mode')
        train()
    elif args.mode == 'eval':
        print('Evaluation Mode')
        eval()
    else:
        print('Mode does not exist')
