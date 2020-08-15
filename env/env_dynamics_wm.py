import numpy as np
import torch
from torch.distributions.categorical import Categorical

import gym
from dynamics.models.mdrnn import MDRNNCell
from dynamics.models.vae import VAE


class EnvDynamics:
    """
    Environment wrapper for model dynamics
    """

    def __init__(self, seed, img_stack, unroll_length, device, c=0.7):
        self.img_stack = img_stack
        self.env = gym.make('CarRacingAdv-v0')
        self.env.seed(seed)
        self.device = device
        # defined as T in paper
        self.unroll_length = unroll_length
        # load models vae and RNN
        self.vae = VAE(3, 32).float().to(self.device)
        self.rnn = MDRNNCell(32, 3, 256, 5).float().to(self.device)
        self.__load_dynamics_params__()
        # init encoder
        self.encoder = self.vae.encoder
        # init decoder
        self.decoder = self.vae.decoder
        # init latent and hidden states
        self.l_state = torch.randn(1, 32)
        self.h_state = 2 * [torch.zeros(1, 256)]
        # init observations/visual observations
        self.obs = None
        self.v_obs = None
        # init latent observations
        self.latent = None
        # Save original state
        self.start_state = self.l_state
        # rendering variables
        self.monitor = None
        self.figure = None

    def __load_dynamics_params__(self):
        vae_weights_file = 'dynamics/param/vae/best.tar'
        rnn_weights_file = 'dynamics/param/mdrnn/best.tar'
        # load vae
        vae_state = torch.load(vae_weights_file, map_location=lambda storage, location: storage)
        print("Loading VAE at epoch {}, with test error {}...".format(vae_state['epoch'], vae_state['precision']))
        self.vae.load_state_dict(vae_state['state_dict'])
        # load rnn
        rnn_state = torch.load(rnn_weights_file, map_location=lambda storage, location: storage)
        print("Loading MDRNN at epoch {}, with test error {}...".format(rnn_state['epoch'], rnn_state['precision']))
        rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
        self.rnn.load_state_dict(rnn_state_dict)

    def __perform_encoder__(self, state):
        state = state.unsqueeze(0)
        self.l_state = self.encoder(state)

    def __perform_decoder__(self):
        self.obs = self.decoder(self.l_state)
        np_obs = self.obs.detach().numpy()
        self.obs = self.obs.clamp(0, 1) * 255
        self.obs = self.obs.permute(0, 2, 3, 1).contiguous().squeeze()
        np_obs = np.clip(np_obs, 0, 1) * 255
        np_obs = np.transpose(np_obs, (0, 2, 3, 1))
        np_obs = np_obs.squeeze()
        np_obs = np_obs.astype(np.uint8)
        self.v_obs = np_obs

    def __perform_rnn__(self, action):
        action = action.unsqueeze(0)
        mu, sigma, pi, r, d, n_h = self.rnn(action, self.l_state, self.h_state)
        pi = pi.squeeze()
        mixt = Categorical(torch.exp(pi)).sample().item()

        self.l_state = mu[:, mixt, :]  # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
        self.h_state = n_h

        self.__perform_decoder__()

    def reset(self):
        self.counter = 0
        self.l_state = self.start_state #torch.randn(1, 32)
        self.h_state = 2 * [torch.zeros(1, 256)]
        # perform decoder step
        self.__perform_decoder__()
        # get first state from random latent state
        assert self.v_obs.shape == (96, 96, 3)
        assert self.obs.shape == (96, 96, 3)
        img_gray = self.rgb2gray(self.obs).unsqueeze(0)
        # Repeat img_gray multiple times
        self.stack = torch.repeat_interleave(img_gray, self.img_stack, dim=0)
        return self.stack

        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((96, 96, 3), dtype=np.uint8))

    def step(self, action, should_unroll=False):
        done = False
        #self.__perform_encoder__(state)
        self.__perform_rnn__(action)
        state_ = self.rgb2gray(self.obs).unsqueeze(0)
        if should_unroll:
            self.counter += 1
        if self.counter == self.unroll_length:
            done = True
            self.counter = 0
        self.stack = torch.cat((self.stack[1:], state_), dim=0)
        assert len(self.stack) == self.img_stack
        return self.stack, 0, done, False

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = torch.matmul(rgb[..., :], torch.tensor([0.299, 0.587, 0.114]))
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    def render(self):  # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((96, 96, 3),
                         dtype=np.uint8))
        self.monitor.set_data(self.v_obs)
        plt.pause(.01)
