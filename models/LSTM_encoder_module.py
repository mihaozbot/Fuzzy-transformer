import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(LSTM_encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        return output