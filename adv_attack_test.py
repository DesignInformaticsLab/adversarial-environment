import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from env.env import Env
from networks.actor_critic import A2CNet

parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = A2CNet(args.img_stack).float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self):
        self.net.load_state_dict(torch.load('param/ppo_net_params.pkl', map_location='cpu'))


if __name__ == "__main__":
    agent = Agent()
    agent.load_param()
    env = Env(args.seed, args.img_stack, args.action_repeat)

    training_records = []
    running_score = 0
    state = env.reset()

    # load adv input
    delta_s = np.load('param/adv.npy')
    # show adv
    fig = plt.figure(figsize=(8, 8))
    plt.title('Stack of ' + str(args.img_stack) + ' adversarial signals seen by Agent')
    plt.axis('off')
    columns, rows = args.img_stack // 2, args.img_stack // 2
    for i in range(1, columns * rows + 1):
        img = delta_s[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()

    for i_ep in range(10):
        score = 0
        state = env.reset()

        for t in range(1000):
            # steps range to render attack in 1000
            attack_render = [30, 40]
            if t in np.arange(attack_render[0], attack_render[1] + 1):
                if t in attack_render:
                    title = 'Attack Started' if t == attack_render[0] else 'Attack ended'
                    title += ' (showing first frame of 4 frames visible to policy)'
                    plt.imshow((state + delta_s)[0], cmap='gray')
                    plt.axis('off')
                    plt.title(title)
                    plt.show()
                state += delta_s

            action = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
