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
from dynamics.models.RNN import RNN
from dynamics.models.UNet import UNet

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Train UNet for reconstruction')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seeds', type=int, nargs='*', default=[0, 88], metavar='N', help='random training seed and testing seed (default: 0 & 88)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs the model needs to be trained')
parser.add_argument('--sample_size', type=float, default=1e4, help='# of frames to collect')
parser.add_argument('--batch_size', type=int, default=10, help='batch size for training the model')
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
HIDDEN_SIZE = 256
LATENT_SIZE = 16
ACTION_SIZE = 3
SEQ_LEN = 32

rnn_weights_file_path = 'dynamics/param/RNN.pkl'
unet_weights_file_path = '/Users/traviszhang/Python Scripts/deeplearning/Science Fair 2019-2020/Project 2/adversarial-environment/dynamics/param/UNet.pkl'
data_file_path = '/Users/traviszhang/Python Scripts/deeplearning/Science Fair 2019-2020/Project 2/adversarial-environment/dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size), args.seeds[0])
test_data_file_path = '/Users/traviszhang/Python Scripts/deeplearning/Science Fair 2019-2020/Project 2/adversarial-environment/dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size), args.seeds[1])

def save_param(net, weights_file_path):
    file_path = weights_file_path
    torch.save(net.state_dict(), file_path)
    print('NN saved in ', file_path)


def load_param(net, weights_file_path):
    if device == torch.device('cpu'):
        net.load_state_dict(torch.load(weights_file_path, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(weights_file_path))

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

def return_sequence(s, next_s, a):
    # Calculate no of sequences to store
    no_of_sequences = int(sample_size/SEQ_LEN)
    train_data = np.zeros((no_of_sequences, SEQ_LEN, 96, 96), dtype=np.float64)
    gt_data = np.zeros((no_of_sequences, SEQ_LEN, 96, 96), dtype=np.float64)
    a_data = np.zeros((no_of_sequences, SEQ_LEN, 3), dtype=np.float64)

    seq_no = 0
    while seq_no < no_of_sequences:
        sequence = []
        gt_sequence = []
        a_sequence = []
        if seq_no < no_of_sequences:
            # Split complete trajectory into sequences of length SEQ_LEN
            for seq in range(int(SEQ_LEN)):
                if seq < SEQ_LEN/2:
                    sequence.append(s[seq + seq_no * SEQ_LEN])
                    sequence.append(next_s[seq + seq_no * SEQ_LEN])
                    gt_sequence.append(s[seq + 1 + seq_no * SEQ_LEN])
                    gt_sequence.append(next_s[seq + 1 + seq_no * SEQ_LEN])

                a_sequence.append(a[seq + seq_no * SEQ_LEN])

            if SEQ_LEN % 2 == 1:
                sequence.append(s[SEQ_LEN - 1])
                gt_sequence.append(SEQ_LEN - 1)
                a_sequence.append(SEQ_LEN - 1)

            train_data[seq_no] = np.array(sequence)
            gt_data[seq_no] = np.array(gt_sequence)
            a_data[seq_no] = np.array(a_sequence)
        else:
            # This portion is to generate the last sequence when sample_size does not divide evenly into SEQ_LEN
            # Still a work in progress
            pass
            # no_of_remaining_frames = sample_size - int(sample_size/SEQ_LEN) * SEQ_LEN
            # for seq in range(int(no_of_remaining_frames/2)):
            #     sequence.append(s[seq + (seq_no - 1) * SEQ_LEN])
            #     sequence.append(next_s[seq + (seq_no - 1) * SEQ_LEN])
            #
            # train_data = np.append(train_data, np.expand_dims(np.array(sequence), axis=0), axis=0)

        seq_no += 1

    return train_data, gt_data, a_data

def loss_fn(pred_latent, gt_latent):
    mse = nn.MSELoss()
    MSE = mse(pred_latent, gt_latent)

    return MSE

def train():
    # Initialize pre-trained policy agent
    agent = Agent(args.img_stack, device)
    agent.load_param()
    # Initialize environments
    env = Env(args.seeds[0], args.img_stack, args.action_repeat)
    test_env = Env(args.seeds[1], args.img_stack, args.action_repeat)
    # Initialize random policy agent
    rand_agent = RandomAgent(args.img_stack, env, args.seeds[0], device)
    test_rand_agent = RandomAgent(args.img_stack, test_env, args.seeds[1], device)
    # Initialize UNet & RNN model
    rnn = RNN(HIDDEN_SIZE, ACTION_SIZE, LATENT_SIZE).to(device)
    unet = UNet().to(device)
    # Initialize optimizer and setup scheduler and earlystopping
    optimizer = optim.Adam(rnn.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    if os.path.isfile(data_file_path):
        print('Trajectory Found. Loading data...')
        rand_buffer = np.load(data_file_path)['arr_0']
        test_rand_buffer = np.load(test_data_file_path)['arr_0']
    else:
        if not os.path.isdir('dynamics/trajectories'):
            os.mkdir('dynamics/trajectories')

        print('Collecting Random Trajectories')
        rand_buffer = collect_trajectories(rand_agent, env, sample_size)
        np.savez_compressed(data_file_path, rand_buffer)
        print('Saved Random Trajectories in ', data_file_path)
        print('Collecting Random Test Trajectories')
        test_rand_buffer = collect_trajectories(test_rand_agent, test_env, sample_size)
        np.savez_compressed(test_data_file_path, test_rand_buffer)
        print('Saved Test Random Trajectories in ', test_data_file_path)

    # Generate sequences
    train_s, gt_s, train_a = return_sequence(rand_buffer['s'][:, 3], rand_buffer['s_'][:, 3], rand_buffer['a'])
    train_s, gt_s, train_a = torch.tensor(train_s, dtype=torch.float).to(device), torch.tensor(gt_s, dtype=torch.float).to(device), torch.tensor(train_a, dtype=torch.float).to(device)

    test_s, test_gt_s, test_a = return_sequence(test_rand_buffer['s'][:, 3], test_rand_buffer['s_'][:, 3], test_rand_buffer['a'])
    test_s, test_gt_s, test_a = torch.tensor(test_s, dtype=torch.float).to(device), torch.tensor(test_gt_s, dtype=torch.float).to(device), torch.tensor(test_a, dtype=torch.float).to(device)

    # Load up UNet weights
    load_param(unet, unet_weights_file_path)

    rnn.train()

    print('Training')
    for i in tqdm.trange(epochs):
        running_loss, no_of_batches = 0, 0
        test_running_loss, test_batch = 0, 0

        for index in BatchSampler(SubsetRandomSampler(range(train_s.shape[0])), batch_size, False):
            # train_s[index] has shape (batch_size, seq_len, channel, height, width)
            # Converts each frame in the sequence to latent vector, while keeping batch the same
            latent_s = torch.empty(len(train_s[index]), SEQ_LEN, 16, dtype=torch.float).to(device)
            latent_gt_s = torch.empty(len(train_s[index]), SEQ_LEN, 16).to(device)
            for j in range(SEQ_LEN):
                _, latent_s[:, j] = unet(train_s[index][:, j].unsqueeze(1)).squeeze()
                _, latent_gt_s[:, j] = unet(gt_s[index][:, j].unsqueeze(1)).squeeze()

            pred_latent_next_s = rnn(train_a[index], latent_s)

            loss = loss_fn(pred_latent_next_s, latent_gt_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            no_of_batches += 1

        running_loss = running_loss / no_of_batches
        print(' loss: ', running_loss)

        with torch.no_grad():
            for index in BatchSampler(SubsetRandomSampler(range(test_s.shape[0])), batch_size, False):
                latent_s = torch.empty(len(test_s[index]), SEQ_LEN, 16, dtype=torch.float).to(device)
                latent_gt_s = torch.empty(len(test_s[index]), SEQ_LEN, 16).to(device)

                for j in range(SEQ_LEN):
                    _, latent_s[:, j] = unet(test_s[index][:, j].unsqueeze(1)).squeeze()
                    _, latent_gt_s[:, j] = unet(test_gt_s[index][:, j].unsqueeze(1)).squeeze()

                pred_latent_next_s = rnn(test_a[index], latent_s)

                test_loss = loss_fn(pred_latent_next_s, latent_gt_s)

                test_running_loss += test_loss.item()
                test_batch += 1

            test_running_loss = test_running_loss / test_batch
            print(' test loss: ', test_running_loss)

        scheduler.step(test_running_loss)
        earlystopping.step(test_running_loss)

        if earlystopping.stop:
            save_param(rnn, rnn_weights_file_path)
            print('Training Stopped Early at epoch ', i)
            break

        if running_loss < 1e-4 or i == epochs - 1:
            save_param(rnn, rnn_weights_file_path)
            break

    print('Training done')

if __name__ == '__main__':
    train()
