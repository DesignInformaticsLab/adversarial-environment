import argparse
import os
from os.path import join, exists
from os import mkdir
import sys
import warnings

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dynamics.vae.utils import RolloutObservationDataset, EarlyStopping, save_checkpoint
from dynamics.vae.vae import VAE

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='Train VAE to encode state space')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs the model needs to be trained')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training the model')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training the model')
parser.add_argument('--log-dir', type=str, help='Directory where results are logged')
parser.add_argument('--no-reload', action='store_true', help='Best model is not reloaded if specified')
parser.add_argument('--no-samples', action='store_true', help='Does not save samples during training if specified')
args = parser.parse_args()

# GPU parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# Hyper Parameters
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
])

dataset_train = RolloutObservationDataset('datasets/carracing/random/brown', transform_train, train=True)
dataset_test = RolloutObservationDataset('datasets/carracing/random/brown', transform_test, train=False)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

data_iter = iter(train_loader)
print(data_iter.next().shape)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    MSE = nn.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return MSE + KLD


def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == '__main__':
    # check vae dir exists, if not, create it
    vae_dir = join(args.log_dir, 'vae')
    if not exists(vae_dir):
        mkdir(vae_dir)
        mkdir(join(vae_dir, 'samples'))

    reload_file = join(vae_dir, 'best.tar')
    if not args.no_reload and exists(reload_file):
        state = torch.load(reload_file)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])

    cur_best = None

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test()
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        # checkpointing
        best_filename = join(vae_dir, 'best.tar')
        filename = join(vae_dir, 'checkpoint.tar')
        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'precision': test_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict()
        }, is_best, filename, best_filename)

        if not args.no_samples:
            with torch.no_grad():
                sample = torch.randn(96, 48).to(device)
                sample = model.decoder(sample).cpu()
                save_image(sample.view(64, 3, 96, 48),
                           join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
