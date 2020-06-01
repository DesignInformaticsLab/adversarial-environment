import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamics.utils import EarlyStopping, load_data, load_param, save_param
from dynamics.models.RNN import RNN
from dynamics.models.UNet import UNet

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Train UNet for reconstruction')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seeds', type=int, nargs='*', default=[0, 88], metavar='N',
                    help='random training seed and testing seed (default: 0 & 88)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs the model needs to be trained')
parser.add_argument('--sample_size', type=float, default=1e4, help='# of frames to collect')
parser.add_argument('--batch_size', type=int, default=20, help='batch size for training the model (use 10x)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training the model')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--hidden-size', type=int, default=256, help='hidden size for RNN model')
parser.add_argument('--seq-len', type=int, default=25, help='sequence length')
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
SEQ_LEN = args.seq_len
HIDDEN_SIZE = args.hidden_size
LATENT_SIZE = 16
ACTION_SIZE = 3

rnn_weights_file_path = 'dynamics/param/RNN.pkl'
unet_weights_file_path = 'dynamics/param/UNet.pkl'
data_file_path = 'dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size), args.seeds[0])
test_data_file_path = 'dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size),
                                                                                           args.seeds[1])


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.shape[0] // bsz
    # get one data point size. In this case, it is (96, 96)
    dp_size = data.shape[1:]
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1, *dp_size).contiguous()
    return data.to(device)


def get_batch(state, action, state_, i):
    seq_len = min(SEQ_LEN, state.size(1) - i)
    state_b = state[:, i:i + seq_len]
    action_b = action[:, i: i + seq_len]
    state__b = state_[:, i: i + seq_len]
    return state_b, action_b, state__b


# As of now this function is not used
def return_sequence(s, next_s, a):
    # Calculate no of sequences to store
    no_of_sequences = int(sample_size / SEQ_LEN)
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
                if seq < SEQ_LEN / 2:
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


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train():
    # Initialize UNet & RNN model
    rnn = RNN(HIDDEN_SIZE, ACTION_SIZE, LATENT_SIZE).to(device)
    unet = UNet().to(device)
    # Initialize optimizer and setup scheduler and early stopping
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    early_stopping = EarlyStopping('min', patience=30)

    rand_buffer = load_data(data_file_path, train=True)
    test_rand_buffer = load_data(test_data_file_path, train=False)

    # Generate sequences
    # train_s, gt_s, train_a = return_sequence(rand_buffer['s'][:, 3], rand_buffer['s_'][:, 3], rand_buffer['a'])
    train_s = batchify(torch.tensor(rand_buffer['s'][:, 3]).float().to(device), batch_size)
    train_a = batchify(torch.tensor(rand_buffer['a']).float().to(device), batch_size)
    train_s_ = batchify(torch.tensor(rand_buffer['s_'][:, 3]).float().to(device), batch_size)

    test_s = batchify(torch.tensor(test_rand_buffer['s'][:, 3]).float().to(device), batch_size)
    test_a = batchify(torch.tensor(test_rand_buffer['a']).float().to(device), batch_size)
    test_s_ = batchify(torch.tensor(test_rand_buffer['s_'][:, 3]).float().to(device), batch_size)

    # Load UNet weights
    load_param(unet, unet_weights_file_path, device)

    epoch_start_time = time.time()
    for epoch in range(1, epochs + 1):
        rnn.train()
        running_loss, no_of_batches = 0, 0
        test_running_loss, test_batch = 0, 0
        hidden = rnn.init_hidden(args.batch_size, device)
        for batch, i in enumerate(range(0, train_s.size(1), SEQ_LEN)):
            s, a, s_ = get_batch(train_s, train_a, train_s_, i)
            # convert to latents
            with torch.no_grad():
                batch_size_tmp, seq_len_tmp = s.shape[:2]
                _, latent_s = unet(state=s.reshape(-1, 1, *s.shape[-2:]))
                latent_s = latent_s.reshape(batch_size_tmp, seq_len_tmp, -1)
                _, latent_s_ = unet(state=s_.reshape(-1, 1, *s.shape[-2:]))
                latent_s_ = latent_s_.reshape(batch_size_tmp, seq_len_tmp, -1)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            rnn.zero_grad()
            hidden = repackage_hidden(hidden)
            output, hidden = rnn(a, latent_s, hidden)
            loss = loss_fn(output, latent_s_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
            for p in rnn.parameters():
                # TO DO: Need to fix this part based on how lr_scheduler works
                p.data.add_(-optimizer.param_groups[len(optimizer.param_groups) - 1]['lr'], p.grad)

            running_loss += loss.item()
            no_of_batches += 1

        running_loss = running_loss / no_of_batches

        # Turn on evaluation mode which disables dropout.
        rnn.eval()
        hidden = rnn.init_hidden(batch_size, device)
        with torch.no_grad():
            for i in range(0, test_s.size(1), SEQ_LEN):
                s, a, s_ = get_batch(test_s, test_a, test_s_, i)
                # convert to latents
                batch_size_tmp, seq_len_tmp = s.shape[:2]
                _, latent_s = unet(state=s.reshape(-1, 1, *s.shape[-2:]))
                latent_s = latent_s.reshape(batch_size_tmp, seq_len_tmp, -1)
                _, latent_s_ = unet(state=s_.reshape(-1, 1, *s.shape[-2:]))
                latent_s_ = latent_s_.reshape(batch_size_tmp, seq_len_tmp, -1)

                output, hidden = rnn(a, latent_s, hidden)
                hidden = repackage_hidden(hidden)
                test_loss = loss_fn(output, latent_s_)
                test_running_loss += test_loss.item()
                test_batch += 1

        test_running_loss = test_running_loss / test_batch

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | test loss {:5.4f} | '
              .format(epoch, (time.time() - epoch_start_time), running_loss, test_running_loss))
        print('-' * 89)

        scheduler.step(test_running_loss)
        early_stopping.step(test_running_loss)

        if early_stopping.stop:
            save_param(rnn, rnn_weights_file_path)
            print('Training Stopped Early at epoch ', epoch)
            break

        if running_loss < 1e-4 or epoch == epochs:
            save_param(rnn, rnn_weights_file_path)
            break

    print('Training done')


if __name__ == '__main__':
    train()
