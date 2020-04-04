import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class ModelDynamics(nn.Module):
    def __init__(self, input_channels, batch_size, device='cpu'):
        super(ModelDynamics, self).__init__()

        channels = input_channels

        self.beta = torch.rand(batch_size, requires_grad=True, dtype=torch.float).to(device)
        self.B_prime = torch.rand((16, batch_size), requires_grad=True, dtype=torch.float).to(device)

        # Initial convolution block
        out_features = 8
        encode = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Encoding
        for _ in range(5):
            out_features *= 2
            encode += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        encode += [nn.Conv2d(256, 16, 3, stride=2, padding=1)]
        self.encode = nn.Sequential(*encode)
        in_features = 16

        # Decoding
        decode = []
        for i in range(6):
            out_features //= 2
            decode += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        decode += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, 1, 7), nn.Tanh()]
        self.decode = nn.Sequential(*decode)

        # Up sample layer
        self.up_sample = nn.Sequential(
            torch.nn.Linear(3, batch_size),
            torch.nn.LeakyReLU()
        )

    def forward(self, state, action):
        # Reconstruction
        # resize to 48x48
        state = F.interpolate(state, (32, 32))
        # down sample through convolution layers
        down = self.encode(state)
        # up sample through transpose convolution layers
        pred_state = self.decode(down)
        # resize to 96x96
        pred_state = F.interpolate(pred_state, (96, 96)).squeeze(1)


        # Dynamics
        big_lmd = torch.diag_embed(torch.exp(-self.beta.pow(2)))
        # Create lambda*phi
        state_dynamics = torch.matmul(big_lmd, down.view(-1, 16))
        # Create B'*u
        action_dynamics = torch.matmul(self.B_prime, action)
        # Upsample action dynamics
        action_dynamics = self.up_sample(action_dynamics)
        # add state and action dynamics
        result = state_dynamics + torch.transpose(action_dynamics, 0, 1)
        # reshape dynamics
        result = result.reshape(-1, 16, 1, 1)
        # dynamics decode
        dyn_state = self.decode(result)
        # resize to output only one image
        #dyn_state = dyn_state.reshape((-1, 198, 198)).unsqueeze(0)
        #dyn_state = dyn_state.reshape((-1, 134, 134)).unsqueeze(0)
        # resize to 96x96
        dyn_state = F.interpolate(dyn_state, (96, 96)).squeeze(1)
        return pred_state, dyn_state


if __name__ == '__main__':
    bs = 10
    state = torch.zeros(bs, 4, 96, 96)
    action = torch.zeros(bs, 3)

    net = ModelDynamics(4, bs)
    a, b = net(state, action)
    print(a.shape, b.shape)