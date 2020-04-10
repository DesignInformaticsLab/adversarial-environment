import torch
from networks.actor_critic import A2CNet


class Agent:
    """
    Agent for selecting actions based on actor-critic network
    """

    def __init__(self, img_stack, device):
        self.net = A2CNet(img_stack).float().to(device)
        self.device = device

    def select_action(self, state):
        # state array contains values with in the range [-1, 0.9921875]
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def select_action_with_grad(self, state):
        state = state.to(self.device).unsqueeze(0)
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

    def __init__(self, img_stack, device):
        self.device = device

    def select_action(self, state):
        action = torch.rand((1, 3))
        action = action.squeeze().cpu().numpy()
        return action
