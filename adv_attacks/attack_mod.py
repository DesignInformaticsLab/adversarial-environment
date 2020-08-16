import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dynamics.models.mdrnn import MDRNNCell
from dynamics.models.vae import VAE
from networks.actor_critic import A2CNet

warnings.filterwarnings("ignore")
torch.manual_seed(10)

# variables for patch attack, Need to move somewhere else later
box_dim = (32, 32)
box_position = (10, 10)
circle_centre = (24, 72)
circle_radius = 40

attack_type = 'general'

def load_nets():
    # load vae and rnn and a2c
    vae = VAE(3, 32).float()
    rnn = MDRNNCell(32, 3, 256, 5).float()
    vae_weights_file = 'dynamics/param/vae/best.tar'
    rnn_weights_file = 'dynamics/param/mdrnn/best.tar'
    # load vae
    vae_state = torch.load(vae_weights_file, map_location=lambda storage, location: storage)
    print("Loading VAE at epoch {}, with test error {}...".format(vae_state['epoch'], vae_state['precision']))
    vae.load_state_dict(vae_state['state_dict'])
    decoder = vae.decoder
    # load rnn
    rnn_state = torch.load(rnn_weights_file, map_location=lambda storage, location: storage)
    print("Loading MDRNN at epoch {}, with test error {}...".format(rnn_state['epoch'], rnn_state['precision']))
    rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
    rnn.load_state_dict(rnn_state_dict)
    # load A2C
    a2c = A2CNet(4).float()
    a2c_weights_file = 'param/ppo_net_params.pkl'
    a2c.load_state_dict(torch.load(a2c_weights_file, map_location=lambda storage, location: storage))

    # eval
    vae.eval()
    decoder.eval()
    rnn.eval()
    a2c.eval()
    return vae, decoder, rnn, a2c

def render(state):  # pylint: disable=arguments-differ
    """ Rendering """
    figure = plt.figure()
    monitor = plt.imshow(state, cmap='gray')
    plt.pause(.01)


def main():
    # params
    # NOTE: ABOVE 0.3 ADV BOUND DOES NOT CONVERGE CORRECTLY
    adv_bound = 0.1
    unroll_length = 150
    lr = 0.05
    epochs = 15
    vae, decoder, rnn, a2c = load_nets()
    # init start state
    start_l_s = torch.randn(1, 32)
    start_s = decoder(start_l_s)
    # reshape
    start_s = start_s.clamp(0, 1) * 255
    start_s = start_s.permute(0, 2, 3, 1).contiguous().squeeze()
    assert start_s.shape == (96, 96, 3)
    # stack start state
    start_s = torch.matmul(start_s[..., :], torch.tensor([0.299, 0.587, 0.114])).reshape(1, 96, 96)
    start_s = start_s / 128. - 1.
    start_s = torch.repeat_interleave(start_s, 4, dim=0).reshape(1, 4, 96, 96)
    # init hidden
    h_state = 2 * [torch.zeros(1, 256)]

    # load target state
    s_t = np.load(
        'adv_attacks/target_samples/target_2.npy')
    s_t = torch.tensor(s_t).float().reshape(1, 4, 96, 96)

    # init d_s
    d_s = torch.rand(unroll_length, 4, 96, 96, requires_grad=True)
    # init optimizer
    lbfgs_optim = optim.LBFGS([d_s], lr=lr)

    summary_path = f'runs/adv_{adv_bound}_{unroll_length}_timesteps'
    if attack_type != 'general':
        summary_path += f'_{box_dim[0]}x{box_dim[1]}'
    writer = SummaryWriter(summary_path)

    mixt_list = []

    # epochs
    for t in range(epochs):
        last_s = []
        def closure():
            mse_loss = 0
            # each epoch, start at beginning hidden state, latent space, and start state
            n_h = h_state
            l_s = start_l_s
            s = start_s
            # unroll lengths
            for i in range(unroll_length):
                print(f'\nEpoch {t} Time {i}')
                # get adv action from controller
                alpha, beta = a2c(torch.clamp(s + d_s[i], -1, 1))[0]
                a = alpha / (alpha + beta)
                print('adv action: ', a)
                # clean action
                alp, bet = a2c(s)[0]
                a_clean = alp / (alp + bet)
                print('clean action', a_clean)
                # scale action
                a = a * torch.tensor([2., 1., 1.]) + torch.tensor([-1., 0., 0.])
                mu, sigma, pi, r, d, n_h = rnn(a, l_s, n_h)
                pi = pi.squeeze()
                mixt = Categorical(torch.exp(pi)).sample().item()
                # Ensures that mixt is sampled stays the same every epoch
                if len(mixt_list) < unroll_length:
                    mixt_list.append(mixt)
                    l_s_ = mu[:, mixt, :]
                else:
                    l_s_ = mu[:, mixt_list[i], :]
                # get next state
                s_ = decoder(l_s_)
                # reshape
                s_ = s_.clamp(0, 1) * 255
                s_ = s_.permute(0, 2, 3, 1).contiguous().squeeze()
                assert s_.shape == (96, 96, 3)
                # stack next state
                s_ = torch.matmul(s_[..., :], torch.tensor([0.299, 0.587, 0.114])).reshape(1, 96, 96)
                s_ = s_ / 128. - 1.
                s_ = torch.cat((s.reshape(4, 96, 96)[1:], s_), dim=0).reshape(1, 4, 96, 96)
                # get and add loss
                loss = nn.MSELoss()(s_, s_t)
                mse_loss += loss

                s = s_
                l_s = l_s_

            lbfgs_optim.zero_grad()
            mse_loss.backward(retain_graph=True)
            last_s.append(s.squeeze())

            return mse_loss

        epoch_loss = lbfgs_optim.step(closure)
        print(f'Epoch {t}: Loss: {epoch_loss / unroll_length}')

        # only allow unmasked for the patch in case of patch attack
        with torch.no_grad():
            # Box attack
            mask = torch.zeros_like(d_s)
            # Generate box mask
            mask[:, :, box_position[0]: box_position[0] + box_dim[0],
            box_position[1]: box_position[1] + box_dim[1]] = 1


            # Circle attack
            # TODO: Still need to test circle attack
            # mask = torch.ones_like(d_s)
            # # Generate circle mask
            # for i in range(mask.shape[1]):
            #     for j in range(mask.shape[2]):
            #         if (circle_centre[0] - i) ** 2 + (circle_centre[1] - j) ** 2 >= circle_radius ** 2:
            #             mask[:, i, j] = 0

            # Restrict perturbation to be only within the mask
            d_s *= mask

            d_s.clamp_(-adv_bound, adv_bound)

        writer.add_image(f'epoch_{t}_t_{unroll_length - 1}', (last_s[len(last_s) - 1] + d_s[unroll_length - 1])[3].unsqueeze(0))
        writer.add_scalar('loss', epoch_loss / unroll_length, t)

    mixt_np = np.array(mixt_list)
    print(d_s.shape)
    print(mixt_np.shape)
    # save delta_s parameters
    adv_dir = 'adv_attacks/perturbations/'
    adv_bound = str(adv_bound)
    file_path = f'{adv_dir}{attack_type}'
    file_path += f'_{adv_bound}_{unroll_length}_timesteps'
    if attack_type != 'general':
        file_path += f'_{box_dim[0]}x{box_dim[1]}'
    mixt_file_path = file_path + '_mixt'
    np.savez_compressed(file_path, d_s.detach().numpy())
    np.savez_compressed(mixt_file_path, mixt_np)
    print('delta_s saved in ', file_path)
    print('mixt values saved in ', mixt_file_path)

    writer.close()


if __name__ == '__main__':
    main()