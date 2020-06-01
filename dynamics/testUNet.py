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
from dynamics.models.UNet import UNet
from dynamics.utils import load_data, load_param

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Train UNet for reconstruction')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seeds', type=int, nargs='*', default=[0, 88], metavar='N',
                    help='random training seed and testing seed (default: 0 & 88)')
parser.add_argument('--sample_size', type=float, default=1e4, help='# of frames to collect')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the model')
args = parser.parse_args()

# GPU parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seeds[1])
if use_cuda:
    torch.cuda.manual_seed(args.seeds[1])

# Hyper Parameters
sample_size = int(args.sample_size)
batch_size = args.batch_size

weights_file_path = 'dynamics/param/UNet.pkl'
test_data_file_path = 'dynamics/trajectories/random-{}-ns-seed-{}-trajectories.npz'.format(int(sample_size),
                                                                                           args.seeds[1])
img_data_file_path = 'dynamics/imgs'


def loss_fn(recon_x, x):
    mse = nn.MSELoss()
    MSE = mse(recon_x.squeeze(1), x.squeeze(1))

    return MSE


def test():
    # Initialize UNet model
    unet = UNet()

    # Make image dir if it does not exist
    if not os.path.isdir(img_data_file_path):
        os.mkdir(img_data_file_path)

    # Load saved weights into model
    load_param(unet, weights_file_path, device)
    # Load test data
    test_rand_buffer = load_data(test_data_file_path, train=False)

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
            recon_images, _ = unet(images_test[index])

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
            recon, _ = unet(images_test[bounds:bounds + batch_size])
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
