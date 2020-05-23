import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import argparse
import warnings
import tqdm
import sys
import os
import matplotlib.pyplot as plt
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.env_adv import Env
from agents.agents import Agent, RandomAgent

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Learn Dynamics Model of Environment')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs the model needs to be trained')
parser.add_argument('--sample_size', type=float, default=5, help='sample size for collect trajectories')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the model')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training the model')
parser.add_argument('--mode', type=str, default='train', help='set mode for VAE')
args = parser.parse_args()

# GPU parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Hyper Parameters
epochs = args.epochs
sample_size = int(args.sample_size)
batch_size = args.batch_size
lr = args.lr
MAX_TRIALS = 20
MAX_FRAMES = 1000

if args.mode == 'test':
    args.seed = 88

weights_file_path = 'dynamics/param/VAE.pkl'
data_file_path = 'dynamics/trajectories/{}-ns-seed-{}-trajectories'.format(2*sample_size, args.seed)
print(data_file_path)


class VAE(nn.Module):

    def __init__(self):

        nn.Module.__init__(self)

        self.c1 = nn.Conv2d(1, 32, 4, stride=2)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 128, 6, stride=2)
        self.c4 = nn.Conv2d(128, 256, 6, stride=2)

        self.mean = nn.Linear(1024, 32)
        self.logvar = nn.Linear(1024, 32)

        self.z_to_vec = nn.Linear(32, 1024)

        self.d1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.d2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.d3 = nn.ConvTranspose2d(64, 32, 7, stride=2)
        self.d4 = nn.ConvTranspose2d(32, 1, 6, stride=3)

    def encode(self, x):

        batch_size = x.shape[0]
        conv_layers = [self.c1, self.c2, self.c3, self.c4]

        for l in conv_layers:
            x = F.relu(l(x))

        x = x.reshape(batch_size, -1)

        means = self.mean(x)
        logvar = self.logvar(x)

        return means, logvar

    def sample(self, batch_size):

        x = torch.randn(batch_size, 32)
        out = self.decode(x)

        return out.detach()

    def forward(self, x):

        means, logvars = self.encode(x)
        z = self.reparametrize(means, logvars)
        recon = self.decode(z)

        return recon, z, means, logvars

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + eps * std

        return sample

    def decode(self, x):

        batch_size = x.shape[0]

        x = F.relu(self.z_to_vec(x))

        deconv_layers = [self.d1, self.d2, self.d3]
        x = x.reshape(batch_size, -1, 1, 1)
        for l in deconv_layers:
            x = F.relu(l(x))

        return torch.sigmoid(self.d4(x))

def collect_trajectories(agent, env, ns):
    s = []
    a = []
    next_s = []

    state = env.reset()

    for _ in range(ns):
        for _ in range(MAX_FRAMES):
            action = agent.select_action(state)
            state_, _, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            s.append(state)
            next_s.append(state_)
            a.append(action)
            state = state_
            if done:
                state = env.reset()
                break

    return s, a, next_s

def save_param(net):
    file_path = weights_file_path
    torch.save(net.state_dict(), file_path)
    print('NN saved in ', file_path)


def load_param(net):
    if device == torch.device('cpu'):
        net.load_state_dict(torch.load(weights_file_path, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(weights_file_path))

def loss_fn(recon_x, x, mu, logvar):
    mse = nn.MSELoss()
    MSE = mse(recon_x.squeeze(1), x.squeeze(1))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE, KLD

def train():
    # Initialize pre-trained policy agent and random agent
    agent = Agent(args.img_stack, device)
    agent.load_param()
    rand_agent = RandomAgent(args.img_stack, device)
    # Initialize environment
    env = Env(args.seed, args.img_stack, args.action_repeat)
    # Initialize dynamics model
    vae = VAE().to(device)
    # Initialize optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Look for trajectory dir if present
    if os.path.isfile(data_file_path):
        print('Trajectory found, Loading data ...')
        file = np.load(data_file_path)
        s = file['s']
        print('Trajectories loaded')
    else:
        # print('Collecting Pre-trained Policy Trajectories')
        # buffer = collect_trajectories(agent, env, sample_size)
        print('Collecting Random Trajectories')
        s, a, next_s = collect_trajectories(rand_agent, env, sample_size)

        np.savez_compressed(data_file_path, s=s, a=a, next_s=next_s)
        print('Trajectories Data saved in ', data_file_path)


    images = torch.tensor(s[:, 3], dtype=torch.float).unsqueeze(1).to(device)

    vae.train()
    # if os.path.isfile(weights_file_path):
    #     load_param(vae)

    print('Training')
    for i in tqdm.trange(epochs):
        running_loss, mse_loss, kld_loss, no_of_batches = 0, 0, 0, 0
        for index in BatchSampler(SubsetRandomSampler(range(2*sample_size)), batch_size, False):
            recon_images, _, mu, logvar = vae(images[index])

            loss, mse, kld = loss_fn(recon_images, images[index], mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            mse_loss += mse.item()
            kld_loss += kld.item()
            no_of_batches += 1

        running_loss = running_loss / no_of_batches
        mse_loss = mse_loss / no_of_batches
        kld_loss = kld_loss / no_of_batches
        print(' loss: {} mse: {} kld: {}'.format(running_loss, mse_loss, kld_loss))

        if running_loss < 1e-4 or i == epochs - 1:
            save_param(vae)
            break

    print('Training done')

def test():
    # Initialize pre-trained policy agent and random agent
    agent = Agent(args.img_stack, device)
    agent.load_param()
    rand_agent = RandomAgent(args.img_stack, device)
    # Initialize environment
    env = Env(args.seed, args.img_stack, args.action_repeat)
    # Initialize dynamics model
    vae = VAE()

    # Look for trajectory dir if present
    if os.path.isfile(data_file_path):
        print('Trajectory found, Loading data ...')
        file = np.load(data_file_path)
        s = file['s']
        print('Trajectories loaded')
    else:
        # print('Collecting Pre-trained Policy Trajectories')
        # buffer = collect_trajectories(agent, env, sample_size)
        print('Collecting Random Trajectories')
        s, a, next_s = collect_trajectories(rand_agent, env, sample_size)

        np.savez_compressed(data_file_path, s=s, a=a, next_s=next_s)
        print('Trajectories Data saved in ', data_file_path)

    images = torch.tensor(s[:, 3], dtype=torch.float)

    vae.eval()
    load_param(vae)

    print('Evaluation')

    running_loss, mse_loss, kld_loss, no_of_batches = 0, 0, 0, 0
    i = 0
    file = open("dynamics/imgs_v2/Losses.txt", "w")
    file.write("Seed {}".format(args.seed))
    for index in BatchSampler(SubsetRandomSampler(range(2 * sample_size)), batch_size, False):
        with torch.no_grad():
            recon_images, _, mu, logvar = vae(images[index])

            loss, mse, kld = loss_fn(recon_images, images[index], mu, logvar)

        running_loss += loss.item()
        mse_loss += mse.item()
        kld_loss += kld.item()
        no_of_batches += 1

        print('loss: {} mse: {} kld: {}'.format(running_loss / no_of_batches, mse_loss / no_of_batches, kld_loss / no_of_batches))
        file.write("\nLoss for Batch {}: {} mse: {} kld: {}".format(i, loss.item(), mse.item(), kld.item()))

        num = np.random.randint(0, batch_size)
        bounds = np.random.randint(0, sample_size - batch_size)
        with torch.no_grad():
            plt.title('Reconstruction')
            recon, _, _ = vae(images[bounds:bounds + batch_size])
            plt.imshow(recon[num].reshape((96, 96)), cmap='gray')
            plt.savefig('dynamics/imgs_v2/{}_Recon.png'.format(i))
        plt.title('Ground Truth')
        plt.imshow(images[bounds:bounds + batch_size][num].reshape((96, 96)), cmap='gray')
        plt.savefig('dynamics/imgs_v2/{}_GT.png'.format(i))

        i += 1

    print('Total Loss: {} Total mse: {} Total kld: {}'.format(running_loss, mse_loss, kld_loss))
    file.write("\nTotal Loss: {}".format(running_loss. mse_loss, kld_loss))
    file.close()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print('Mode does not exist')
