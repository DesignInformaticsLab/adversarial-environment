import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelDynamics(nn.Module):
    def __init__(self, input_channels, batch_size, device='cpu'):
        super(ModelDynamics, self).__init__()

        channels = input_channels

        self.beta = torch.rand(batch_size, requires_grad=True, dtype=torch.float).to(device)
        self.B_prime = torch.rand((1024, batch_size), requires_grad=True, dtype=torch.float).to(device)

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

        self.encode = nn.Sequential(*encode)

        # Decoding
        decode = []
        for i in range(6):
            out_features //= 2
            decode += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        decode += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.decode = nn.Sequential(*decode)

        # Up sample layer
        self.up_sample = nn.Sequential(
            torch.nn.Linear(3, batch_size),
            torch.nn.LeakyReLU()
        )

        self.up_sample2 = nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU()
        )

        self.down_sample = nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU()
        )

    def forward(self, state, action):
        # Reconstruction
        # resize to 48x48
        state = F.interpolate(state, (48, 48))
        # down sample through convolution layers
        down = self.encode(state)
        # Reduce to latent vector 128x1
        down = self.down_sample(down.view((-1, 1, 1024)))
        # Upsample to image view
        down = self.up_sample2(down)
        # up sample through transpose convolution layers
        pred_state = self.decode(down.view(-1, 256, 2, 2))
        # resize to output only one image
        #pred_state = pred_state.reshape((-1, 198, 198)).unsqueeze(0)
        pred_state = pred_state.reshape((-1, 134, 134)).unsqueeze(0)
        # resize to 96x96
        pred_state = F.interpolate(pred_state, (96, 96)).squeeze(0)


        # Dynamics
        big_lmd = torch.diag_embed(torch.exp(-self.beta.pow(2)))
        # Create lambda*phi
        state_dynamics = torch.matmul(big_lmd, down.view(-1, 256 * 2 * 2))
        # Create B'*u
        action_dynamics = torch.matmul(self.B_prime, action)
        # Upsample action dynamics
        action_dynamics = self.up_sample(action_dynamics)
        # add state and action dynamics
        result = state_dynamics + torch.transpose(action_dynamics, 0, 1)
        # reshape dynamics
        result = result.reshape(-1, 256, 2, 2)
        # dynamics decode
        dyn_state = self.decode(result)
        # resize to output only one image
        #dyn_state = dyn_state.reshape((-1, 198, 198)).unsqueeze(0)
        dyn_state = dyn_state.reshape((-1, 134, 134)).unsqueeze(0)
        # resize to 96x96
        dyn_state = F.interpolate(dyn_state, (96, 96)).squeeze(0)
        return pred_state, dyn_state

class AutoEncoder(nn.Module):
    def __init__(self, batch_size, device='cpu'):
        super(AutoEncoder, self).__init__()

        self.beta = torch.rand(batch_size, requires_grad=True, dtype=torch.float).to(device)
        self.B_prime = torch.rand((128, batch_size), requires_grad=True, dtype=torch.float).to(device)

        self.encode = nn.Sequential(
            torch.nn.Linear(9216, 4608),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4608, 2304),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2304, 1152),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1152, 576),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(576, 128),
            torch.nn.LeakyReLU()
        )

        self.decode = nn.Sequential(
            torch.nn.Linear(128, 576),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(576, 1152),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1152, 2304),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2304, 4608),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4608, 9216),
            torch.nn.LeakyReLU()
        )

        self.up_sample = nn.Sequential(
            torch.nn.Linear(3, batch_size),
            torch.nn.LeakyReLU()
        )

    def forward(self, state, action):
        # Reconstruction
        # Reduce image to size (4, 48, 48)
        state = F.interpolate(state, (48, 48))
        # Flatten state
        state = state.view((-1, 4 * 48 * 48))
        # Encode state to latent vector (128, 1)
        latent = self.encode(state)
        # Decode latent vector
        pred_state = self.decode(latent)
        # Reshape to image size
        pred_state = pred_state.view((-1, 96, 96))


        # Dynamics
        big_lmd = torch.diag_embed(torch.exp(-self.beta.pow(2)))
        # Create lambda*phi
        state_dynamics = torch.matmul(big_lmd, latent)
        # Create B'*u
        action_dynamics = torch.matmul(self.B_prime, action)
        # Upsample action dynamics
        action_dynamics = self.up_sample(action_dynamics)
        # add state and action dynamics
        result = state_dynamics + torch.transpose(action_dynamics, 0, 1)
        # decode result
        dyn_state = self.decode(result)
        # Reshape decoding to image size
        dyn_state = dyn_state.view((-1, 96, 96))

        return pred_state, dyn_state


if __name__ == '__main__':
    bs = 10
    state = torch.zeros(bs, 4, 96, 96)
    action = torch.zeros(bs, 3)

    net = ModelDynamics(4, bs)
    a, b = net(state, action)
    print(a.shape, b.shape)

    net1 = AutoEncoder(bs)
    c, d = net1(state, action)
    print(c.shape, d.shape)
