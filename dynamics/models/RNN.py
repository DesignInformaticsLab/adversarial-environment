import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, hidden_size, action_size, latent_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(latent_size + action_size, hidden_size)

        self.convert_to_latent = nn.Linear(hidden_size, latent_size)

    def forward(self, actions, latents):
        inputs = torch.cat([actions, latents], dim=-1)
        next_hidden, _ = self.rnn(inputs)
        next_latent = self.convert_to_latent(next_hidden)

        return next_latent
