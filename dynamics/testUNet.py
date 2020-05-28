import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import argparse
import warnings
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.env_adv import Env
from agents.agents import Agent, RandomAgent
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
torch.manual_seed(args.seeds[1])
if use_cuda:
    torch.cuda.manual_seed(args.seeds[1])

# Hyper Parameters
epochs = args.epochs
sample_size = int(args.sample_size)
batch_size = args.batch_size
lr = args.lr

weights_file_path = 'dynamics/param/UNet.pkl'
test_data_file_path = 'dynamics/trajectories/{}-ns-seed-{}-trajectories.npz'.format(int(sample_size), args.seeds[1])
img_data_file_path = 'dynamics/imgs'

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

def test():
    # Initialize pre-trained policy agent and random agent
    agent = Agent(args.img_stack, device)
    agent.load_param()
    rand_agent = RandomAgent(args.img_stack, device)
    # Initialize environments
    env = Env(args.seeds[0], args.img_stack, args.action_repeat)
    test_env = Env(args.seeds[1], args.img_stack, args.action_repeat)
    # Initialize UNet model
    unet = UNet()

    # Make image dir if it does not exist
    if not os.path.isdir(img_data_file_path):
        os.mkdir(img_data_file_path)

    # Load saved weights into model
    load_param(unet)

    if os.path.isfile(test_data_file_path):
        print('Trajectory Found. Loading data...')
        test_rand_buffer = np.load(test_data_file_path)['arr_0']
    else:
        print('Collecting Test Trajectories')
        test_rand_buffer = collect_trajectories(agent, test_env, sample_size)
        np.savez_compressed(test_data_file_path, test_rand_buffer)
        print('Saved Test Random Trajectories in ', test_data_file_path)

    # Convert Trajectories to tensor
    images_test = torch.tensor(test_rand_buffer['s_'][:, 3], dtype=torch.float).unsqueeze(1)

    unet.eval()

    file = open(os.path.join(img_data_file_path, "Losses.txt"), "w")
    file.write("Seed {}".format(args.seeds[1]))

    print('Evaluation')
    i = 0
    test_running_loss, test_batch = 0, 0
    for index in BatchSampler(SubsetRandomSampler(range(images_test.shape[0])), batch_size, False):
        with torch.no_grad():
            recon_images = unet(images_test[index])

            test_loss = loss_fn(recon_images, images_test[index])

        test_running_loss += test_loss.item()
        test_batch += 1

        test_running_loss = test_running_loss / test_batch

        print(' test loss: ', test_running_loss)
        file.write("\nLoss for Batch {}: {}".format(i, test_loss.item()))

        # Choose random batch and random image within that batch for visualization
        num = np.random.randint(0, batch_size)
        bounds = np.random.randint(0, sample_size - batch_size)
        with torch.no_grad():
            plt.title('Predicted')
            recon = unet(images_test[bounds:bounds + batch_size])
            plt.imshow(recon[num].reshape((96, 96)), cmap='gray')
            plt.savefig(os.path.join(img_data_file_path, '{}_Recon.png'.format(i)))
        plt.title('Ground Truth current')
        plt.imshow(images_test[bounds:bounds + batch_size][num, :, :].reshape((96, 96)), cmap='gray')
        plt.savefig(os.path.join(img_data_file_path, '{}_GT.png'.format(i)))

        i += 1

    print('Total Loss: {}'.format(test_running_loss))
    file.write("\nTotal Loss: {}".format(test_running_loss))
    file.close()


if __name__ == '__main__':
    test()
