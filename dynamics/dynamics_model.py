import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, image_stack):
        super(DynamicsModel, self).__init__()
        channels = image_stack

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

        # next state prediction network as per koopman's theory
        self.big_lambda = nn.Linear(16, 16, bias=False)
        self.b_prime = nn.Linear(3, 16, bias=False)

        # Decoding
        in_features = 16
        decode = []
        for i in range(6):
            out_features //= 2
            decode += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        decode += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, 1, 7), nn.Tanh()]
        self.decode = nn.Sequential(*decode)

    def encode_state(self, state):
        # reduce input state size
        state = F.interpolate(state, (32, 32))
        # encode NN
        encode = self.encode(state)
        assert encode.shape[1] == 16
        return encode

    def decode_state(self, state):
        # decode NN
        decode = self.decode(state)
        # reduce state size to actual 96x96
        decode = F.interpolate(decode, (96, 96))
        return decode

    def reconstruct(self, state):
        # reconstruct based on encoder and decoder
        return self.decode_state(self.encode_state(state))

    def next_state(self, state, action):
        # encode state
        encode = self.encode_state(state)
        # Flatten
        encode = encode.view(-1, 16)
        # perform multiplication of encoded state with big_lambda
        state_dyn = self.big_lambda(encode)
        # perform multiplication of action with b_prime
        action_dyn = self.b_prime(action)
        assert state_dyn.shape[1] == 16
        assert action_dyn.shape[1] == 16
        # add state and action
        next_state_enc = state_dyn + action_dyn
        # Unflatten
        next_state_enc = next_state_enc.view(-1, 16, 1, 1)
        # decode state
        next_state = self.decode_state(next_state_enc)
        return next_state

    def forward(self, state, action):
        # Reconstruction
        pred_s_t = self.reconstruct(state)
        # Next state prediction
        pred_s_t_plus_1 = self.next_state(state, action)
        return pred_s_t, pred_s_t_plus_1
