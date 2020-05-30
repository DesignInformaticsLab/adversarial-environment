import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agents import Agent, RandomAgent
from env.env_adv import Env, EnvRandom
from dynamics.models.UNet import UNet

parser = argparse.ArgumentParser(description='Show reconstruction of CarRacingAdv environment')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--policy', type=str, choices=['random', 'pretrained'], help='Policy to be chosen for agent')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--save', action='store_true', help='save the video')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


def get_recon(net, img):
    with torch.no_grad():
        tensor_img = torch.tensor(img).float().to(device).reshape((1, 1, 96, 96))
        tensor_recon = net(tensor_img)
    return tensor_recon.squeeze().cpu().numpy


if __name__ == "__main__":
    # Initialize environment and agent
    env, agent = None, None
    if args.policy == 'random':
        env = EnvRandom(args.seed, args.img_stack, args.action_repeat)
        agent = RandomAgent(env, args.seed, device)
        agent.generate_actions()
    elif args.policy == 'pretrained':
        env = Env(args.seed, args.img_stack, args.action_repeat)
        agent = Agent(args.img_stack, device)
        agent.load_param()
    # load VAE
    vae = UNet().to(device)
    vae.eval()
    weights_file_path = 'dynamics/param/UNet.pkl'
    if device == torch.device('cpu'):
        vae.load_state_dict(torch.load(weights_file_path, map_location='cpu'))
    else:
        vae.load_state_dict(torch.load(weights_file_path))

    score = 0
    state = env.reset()

    # store original and reconstructed images
    gt_list, recon_list = [], []
    gt_list.append(state[0].squeeze())
    recon_list.append(get_recon(vae, state[0]))

    # Loop over frames of one episode
    print('Running environment to collect frames')
    for t in range(1000):
        action = agent.select_action(state)
        state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        # store original and reconstructed images
        gt_list.append(state_[0].squeeze())
        recon_list.append(get_recon(vae, state_[0]))
        score += reward
        state = state_
        if done or die:
            break

    print('Finished episode. Score: {:.2f}\t'.format(score))

    # Initialize subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    gt_img = axes[0].imshow(gt_list[0], cmap='gray', animated=True)
    axes[0].axis('off')
    axes[0].set_title('Original')
    recon_img = axes[1].imshow(recon_list[0], cmap='gray', animated=True)
    axes[1].axis('off')
    axes[1].set_title('Reconstructed')


    def update_fig(i):
        # function to update fig
        if i < len(gt_list):
            gt_img.set_array(gt_list[i])
            recon_img.set_array(recon_list[i])
        # return the artists set
        return [gt_img, recon_img]


    # kick off the animation
    ani = animation.FuncAnimation(fig, update_fig, interval=150, blit=True)
    if args.save:
        ani.save('dynamics/imgs/reconstruction.gif')
    plt.show()
