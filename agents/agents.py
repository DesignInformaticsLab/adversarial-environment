import torch
from networks.actor_critic import A2CNet
import numpy as np
import math


class Agent:
    """
    Agent for selecting actions based on actor-critic network
    """

    def __init__(self, img_stack, device):
        self.net = A2CNet(img_stack).float().to(device)
        self.device = device

    def select_action_np(self, state):
        # state array contains values with in the range [-1, 0.9921875]
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def select_action(self, state):
        # state array contains values with in the range [-1, 0.9921875]
        with torch.no_grad():
            alpha, beta = self.net(state.to(self.device).unsqueeze(0))[0]
        action = alpha / (alpha + beta)

        action = action.squeeze()#.cpu().numpy()
        return action

    def select_action_with_grad(self, state):
        state = state.to(self.device).float().unsqueeze(0)
        alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)
        return action.squeeze()

    def load_param(self):
        if self.device == torch.device('cpu'):
            self.net.load_state_dict(torch.load('param/ppo_net_params.pkl', map_location='cpu'))
        else:
            self.net.load_state_dict(torch.load('param/ppo_net_params.pkl'))


class RandomAgent:
    """
    Agent for selecting actions based on random policy
    """

    def __init__(self, env, seed, device, seq_len=1000):
        self.device = device
        self.env = env
        self.seq_len = seq_len
        np.random.seed(seed)

    def generate_actions(self, noise_type='brown'):
        if noise_type == 'white':
            self.action_seq = [self.env.env.action_space.sample() for _ in range(self.seq_len)]
        elif noise_type == 'brown':
            self.action_seq = self._sample_continuous_policy(self.env.env.action_space, self.seq_len, 1. / 50)

    def _sample_continuous_policy(self, action_space, seq_len, dt):
        """ Sample a continuous policy.
        Atm, action_space is supposed to be a box environment. The policy is
        sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).
        :args action_space: gym action space
        :args seq_len: number of actions returned
        :args dt: temporal discretization
        :returns: sequence of seq_len actions
        """
        actions = [action_space.sample()]
        for _ in range(seq_len):
            daction_dt = np.random.randn(*actions[-1].shape)
            actions.append(np.clip(actions[-1] + math.sqrt(dt) * daction_dt, action_space.low, action_space.high))
        return actions

    def select_action(self, state):
        action = self.action_seq[0]
        self.action_seq.pop(0)
        return action