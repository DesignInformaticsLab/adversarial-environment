import gym
import numpy as np
import torch
from dynamics.dynamics_model import DynamicsModel


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
        # load model dynamics
        self.dynamics = DynamicsModel(self.img_stack).float().to(self.device)
        self.__load_dynamics_params__(c, device)

    def __load_dynamics_params__(self, c, device):
        weights_file = f'dynamics/param/learn_dynamics_lmd_{c}.pkl'
        if device == torch.device('cpu'):
            self.dynamics.load_state_dict(torch.load(weights_file, map_location='cpu'))
        else:
            self.dynamics.load_state_dict(torch.load(weights_file))

    def reset(self):
        self.counter = 0
        img_rgb = self.env.reset()
        img_gray = torch.tensor(self.rgb2gray(img_rgb)).float()
        self.stack = [img_gray] * self.img_stack
        return torch.stack(self.stack).float().to(self.device)

    def step(self, state, action, should_unroll=False):
        done = False
        s = state.unsqueeze(0)
        a = action.unsqueeze(0)
        _, state_ = self.dynamics(s, a)
        state_ = state_.squeeze()
        if should_unroll:
            self.counter += 1
        if self.counter == self.unroll_length:
            done = True
            self.counter = 0
        self.stack.pop(0)
        self.stack.append(state_)
        assert len(self.stack) == self.img_stack
        return torch.stack(self.stack).float().to(self.device), 0, done, False

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray
