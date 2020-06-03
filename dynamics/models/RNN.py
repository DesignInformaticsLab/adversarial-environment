import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, hidden_size, action_size, latent_size, n_layers=2, drop_prob=0.5):
        super(RNN, self).__init__()
        # define output size, number of layers, hidden size
        self.output_size = latent_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # define LSTM
        self.rnn = nn.LSTM(latent_size + action_size, hidden_size, n_layers, batch_first=True)
        # define dropout if necessary during training. Use drop_prob = 0 for no dropout
        self.dropout = nn.Dropout(drop_prob)
        # Fully connected layer to convert hidden to output
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, actions, latents, hidden=None):
        # get batch_size and seq_len from tensor
        batch_size, seq_len = actions.shape[:2]
        # concatenate states and actions
        inputs = torch.cat([actions, latents], dim=-1)
        # apply RNN layers
        out, hidden = self.rnn(inputs, hidden)
        # apply FC layer
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.view(batch_size, seq_len, -1)
        # return output, hidden
        return out, hidden

    def init_hidden(self, batch_size, device):
        # initialize hidden state
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden
