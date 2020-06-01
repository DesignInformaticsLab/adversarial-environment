import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        channels = 1

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

    def forward(self, state=None, latent=None):
        if state is not None:
            # Encode state to the latent space
            return self.encode_state(state)
        if latent is not None:
            # Decode
            return self.decode_state(latent)
