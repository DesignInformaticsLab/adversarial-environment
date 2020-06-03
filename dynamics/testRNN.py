import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamics.utils import load_data, load_param
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
test_data_file_path = 'dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size),
                                                                                           args.seeds[1])
img_data_file_path = 'dynamics/imgs'


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


def test():
    # Initialize UNet & RNN model
    rnn = RNN(HIDDEN_SIZE, ACTION_SIZE, LATENT_SIZE).to(device)
    unet = UNet().to(device)

    test_rand_buffer = load_data(test_data_file_path, train=False)

    # Generate sequences
    test_s = batchify(torch.tensor(test_rand_buffer['s'][:, 3]).float().to(device), batch_size)
    test_a = batchify(torch.tensor(test_rand_buffer['a']).float().to(device), batch_size)
    test_s_ = batchify(torch.tensor(test_rand_buffer['s_'][:, 3]).float().to(device), batch_size)

    # Load UNet and RNN weights
    load_param(unet, unet_weights_file_path, device)
    load_param(rnn, rnn_weights_file_path, device)

    # Make image dir if it does not exist
    if not os.path.isdir(img_data_file_path):
        os.mkdir(img_data_file_path)

    file = open(os.path.join(img_data_file_path, "Losses.txt"), "w")
    file.write("Seed {}".format(args.seeds[1]))

    print('Evaluation')
    epoch_start_time = time.time()
    test_running_loss, test_batch = 0, 1
    # Turn on evaluation mode which disables dropout.
    rnn.eval()
    # Choose random image within that batch for visualization
    index = np.random.randint(0, batch_size)
    with torch.no_grad():
        for batch, i in enumerate(range(0, test_s.size(1), SEQ_LEN)):
            s, a, s_ = get_batch(test_s, test_a, test_s_, i)
            # convert to latents
            batch_size_tmp, seq_len_tmp = s.shape[:2]
            _, latent_s = unet(state=s.reshape(-1, 1, *s.shape[-2:]))
            latent_s = latent_s.reshape(batch_size_tmp, seq_len_tmp, -1)
            _, latent_s_ = unet(state=s_.reshape(-1, 1, *s.shape[-2:]))
            latent_s_ = latent_s_.reshape(batch_size_tmp, seq_len_tmp, -1)

            output, hidden = rnn(a, latent_s)
            # hidden = repackage_hidden(hidden)
            test_loss = loss_fn(output.squeeze(), latent_s_)
            test_running_loss += test_loss.item()
            test_batch += 1

            print('-' * 89)
            print('| end of batch {:3d} | time: {:5.2f}s | test loss {:5.4f} | '
                  .format(batch, (time.time() - epoch_start_time), test_loss.item()))
            print('-' * 89)
            file.write("\nLoss for Batch {}: {}".format(batch, test_loss.item()))

            for j in range(SEQ_LEN):
                plt.title('Predicted')
                pred = unet(latent=output[:, j].reshape(-1, LATENT_SIZE, 1, 1))
                plt.imshow(pred[index].reshape((96, 96)), cmap='gray')
                plt.savefig(os.path.join(img_data_file_path, '{}_Pred.png'.format(j + SEQ_LEN*batch)))
                plt.title('Ground Truth current')
                plt.imshow(s_[index, j].reshape((96, 96)), cmap='gray')
                plt.savefig(os.path.join(img_data_file_path, '{}_GT.png'.format(j + SEQ_LEN*batch)))

    test_running_loss = test_running_loss / test_batch

    print('total test loss: {:5.4f}'.format(test_running_loss))


if __name__ == '__main__':
    test()
