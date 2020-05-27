import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import argparse
import warnings
import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.env_adv import Env
from agents.agents import Agent, RandomAgent
from dynamics.utils import EarlyStopping, ReduceLROnPlateau
from dynamics.models.UNet import UNet

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Train UNet for reconstruction')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seeds', type=int, nargs='*', default=[0, 88], metavar='N', help='random training seed and testing seed (default: 0 & 88)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs the model needs to be trained')
parser.add_argument('--sample_size', type=float, default=1e4, help='# of frames to collect')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the model')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training the model')
args = parser.parse_args()

# GPU parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seeds[0])
if use_cuda:
    torch.cuda.manual_seed(args.seeds[0])

# Hyper Parameters
epochs = args.epochs
sample_size = int(args.sample_size)
batch_size = args.batch_size
lr = args.lr

weights_file_path = 'dynamics/param/UNet.pkl'
data_file_path = 'dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size), args.seeds[0])
test_data_file_path = 'dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size), args.seeds[1])

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

def loss_fn(recon_x, x):
    mse = nn.MSELoss()
    MSE = mse(recon_x.squeeze(1), x.squeeze(1))

    return MSE

def train():
    # Initialize pre-trained policy agent and random agent
    agent = Agent(args.img_stack, device)
    agent.load_param()
    rand_agent = RandomAgent(args.img_stack, device)
    # Initialize environment
    env = Env(args.seeds[0], args.img_stack, args.action_repeat)
    test_env = Env(args.seeds[1], args.img_stack, args.action_repeat)
    # Initialize dynamics model
    unet = UNet().to(device)
    # Initialize optimizer and setup scheduler and earlystopping
    optimizer = optim.Adam(unet.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)


    if os.path.isfile(data_file_path):
        rand_buffer = np.load(data_file_path)['arr_0']
        test_rand_buffer = np.load(test_data_file_path)['arr_0']
    else:
        print('Collecting Random Trajectories')
        rand_buffer = collect_trajectories(rand_agent, env, sample_size)
        np.savez_compressed(data_file_path, rand_buffer)
        print('Saved Random Trajectories in ', data_file_path)
        print('Collecting Random Test Trajectories')
        test_rand_buffer = collect_trajectories(rand_agent, test_env, sample_size)
        np.savez_compressed(test_data_file_path, test_rand_buffer)
        print('Saved Test Random Trajectories in ', test_data_file_path)

    # Convert Trajectories to tensor
    images = torch.tensor(rand_buffer['s_'][:, 3], dtype=torch.float).unsqueeze(1).to(device)
    images_test = torch.tensor(test_rand_buffer['s_'][:, 3], dtype=torch.float).unsqueeze(1).to(device)

    unet.train()

    print('Training')
    for i in tqdm.trange(epochs):
        running_loss, no_of_batches = 0, 0
        test_running_loss, test_batch = 0, 0
        for index in BatchSampler(SubsetRandomSampler(range(len(images))), batch_size, False):
            recon_images, _ = unet(images[index])

            loss = loss_fn(recon_images, images[index])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            no_of_batches += 1

        running_loss = running_loss / no_of_batches
        print(' loss: ' , running_loss)

        with torch.no_grad():
            for index in BatchSampler(SubsetRandomSampler(range(len(images_test))), batch_size, False):
                recon_images, _ = unet(images_test[index])

                test_loss = loss_fn(recon_images, images[index])

                test_running_loss += test_loss.item()
                test_batch += 1

            test_running_loss = test_running_loss / test_batch

            print(' test loss: ', test_running_loss)

        scheduler.step(test_running_loss)
        earlystopping.step(test_running_loss)

        if earlystopping.stop:
            save_param(unet)
            print('Training Stopped Early at epoch ', i)
            break

        if running_loss < 1e-4 or i == epochs - 1:
            save_param(unet)
            break

    print('Training done')


if __name__ == '__main__':
    train()
