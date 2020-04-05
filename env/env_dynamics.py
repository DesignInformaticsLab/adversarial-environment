import gym
import numpy as np
import torch
from dynamics.dynamics_model import DynamicsModel


class EnvDynamics:
    """
    Environment wrapper for model dynamics
    """

    def __init__(self, seed, img_stack, device_name, c=0.7):
        self.img_stack = img_stack
        self.env = gym.make('CarRacingAdv-v0')
        self.env.seed(seed)
        self.device = torch.device(device_name)
        # defined as T in paper
        self.unroll_length = 20
        # load model dynamics
        self.dynamics = DynamicsModel(self.img_stack, 50, device_name).to(self.device)
        self.__load_dynamics_params__(c, device_name)

    def __load_dynamics_params__(self, c, device_name):
        weights_file = f'dynamics_model/param/learn_dynamics_lmd_{c}.pkl'
        if device_name == 'cpu':
            self.dynamics.load_state_dict(torch.load(weights_file, map_location='cpu'))
        else:
            self.dynamics.load_state_dict(torch.load(weights_file))

    def reset(self):
        self.counter = 0
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        return np.array(self.stack)

    def step(self, state, action):
        done = False
        s = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        a = torch.from_numpy(action).float().to(self.device).unsqueeze(0)
        print(s.shape, a.shape)
        _, state_ = self.dynamics(s, a)
        self.counter += 1
        if self.counter == self.unroll_length:
            done = True
            self.counter = 0
        self.stack.pop(0)
        self.stack.append(state_)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), 0, done, False

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray
