import torch
import torch.optim as optim
import numpy as np
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

    # initialize perturbation signal
    delta_s = np.random.random(state.shape) * 0.001
    # create a buffer to state and delta_ts
    buffer_capacity = 128
    buffer_type = np.dtype(
        [('s', np.float64, (args.img_stack, 96, 96)), ('d_s', np.float64, (args.img_stack, 96, 96))])
    buffer = np.empty(buffer_capacity, dtype=buffer_type)
    buffer_counter, is_buffer_full = 0, False

    # initialize and load network
    net = A2CNet(args.img_stack).double().to(device)
    net.load_state_dict(torch.load('param/ppo_net_params.pkl', map_location='cpu'))

    for i_ep in range(50):
        score = 0
        state = env.reset()

        for t in range(1000):
            action = agent.select_action(state)
            # add state and delta_s to buffer
            buffer[buffer_counter] = (state, delta_s)
            buffer_counter += 1
            # check if the buffer is filled
            if buffer_counter == buffer_capacity:
                buffer_counter = 0
                is_buffer_full = True
            if is_buffer_full:
                # optimize perturbation

                # get states from the buffer
                s = torch.tensor(buffer['s'], dtype=torch.double).to(device)
                # iterate through 10 epochs
                for _ in range(10):
                    # get d_s (perturbation from buffer), d_s shape ~ (128, 4, 96, 96)
                    d_s = torch.tensor(buffer['d_s'], dtype=torch.double).to(device)
                    # set grad true so that it can be differentiable
                    d_s.requires_grad = True

                    # get actions and value functions from NN based  on s + d_s.
                    (alpha, beta), v_adv = net(s + d_s)
                    a_adv = alpha / (alpha + beta)

                    # take a left turn as target action [0.25, 0.5, 0.25]. This can be changed
                    target_a = torch.from_numpy(np.array([0.25, 0.5, 0.25], dtype=np.double))
                    # mean squuare loss
                    mse_loss = ((a_adv - target_a) ** 2).sum()
                    # set up adam optimizer with d_s (perturbation) as parameters
                    mse_optim = optim.Adam([d_s], lr=0.01)
                    # set gradients zero before back propagation
                    mse_optim.zero_grad()
                    # perform back propagation
                    mse_loss.backward()
                    mse_optim.step()

                    # save delta_s as average of d_s in the entire batch
                    delta_s = np.average(d_s.detach().numpy(), axis=0)
                    # print delta_s and mse loss
                    mse_loss_scalar = mse_loss.detach().numpy().item() / buffer_capacity
                    print('Sum of delta_s:', np.sum(delta_s), ', MSE Loss: ', mse_loss_scalar)
                    # check if mse loss is less than 10e-3, then save delta_s param
                    if mse_loss_scalar < 10e-3:
                        np.save('param/adv', delta_s)
                        print('delta_s saved')
                        exit(0)

                    # update buffer with new delta_s
                    for i in range(buffer_capacity):
                        buffer[i]['d_s'] = delta_s

            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            score += reward
            state = state_
            if done or die:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
