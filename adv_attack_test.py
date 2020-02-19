import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from env.env import Env
from agents.agent import Agent

parser = argparse.ArgumentParser(description='Adversarial attacks on the CarRacing-v0 environment')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--attack_type', type=str, default='general', metavar='N', help='type of the attack')
# only use if attack_type is not general
parser.add_argument('--patch_type', type=str, default='box', metavar='N', help='type of patch in patch attack type')
parser.add_argument('--patch_size', type=int, default=24, metavar='N', help='size of patch in patch attack type')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


def test_attack():
    agent = Agent(args.img_stack, device)
    agent.load_param(device)
    env = Env(args.seed, args.img_stack, args.action_repeat)

    # load adv input, by default general attack perturbation
    delta_s = np.load('param/adv_general.npy')
    if args.attack_type != 'general':
        file_path = 'param/adv_' + args.attack_type
        if args.attack_type == 'patch':
            file_path += '_' + args.patch_type
        file_path += '.npy'
        delta_s = np.load(file_path)
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

            action = agent.select_action(state, device)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))


if __name__ == "__main__":
    test_attack()
