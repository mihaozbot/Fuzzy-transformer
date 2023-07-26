import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, regressor_dim, sequence_length, ar_order):
        super(LSTMDecoder, self).__init__()

        self.regressor_dim = regressor_dim
        self.sequence_length = sequence_length
        self.ar_order = ar_order
        self.hidden_dim = ar_order  # set hidden_dim to be the same as ar_order

         # Initialize the LSTM, set batch_first=True if your input is of shape (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(1, self.hidden_dim, batch_first=True)

        # Linear layer to map the hidden state of LSTM to output
        self.linear = nn.Linear(self.hidden_dim, 1)

        # Layers to initialize the hidden and cell states
        self.init_h = nn.Linear(self.regressor_dim, self.hidden_dim)
        self.init_c = nn.Linear(self.regressor_dim, self.hidden_dim)

    def forward(self, y, u):
        batch_size = y.size(0)
        device = y.device

        # Start by filling in the initial conditions for the output
        output = y

        # Initialize hidden and cell states using the condensed information
        h_t = self.init_h(u).unsqueeze(0)
        c_t = self.init_c(u).unsqueeze(0)

        for t in range(self.ar_order, self.sequence_length+self.ar_order):
            # Use the most recent output values and exogenous variables as the regressor
            regressor = output[:, (t-self.ar_order):t]

            # Pass the regressor through the LSTM
            lstm_out, (h_t, c_t) = self.lstm(regressor.unsqueeze(2), (h_t, c_t))

            # Compute the prediction using the last LSTM output
            prediction_t = self.linear(lstm_out[:, -1, :])

            # Concatenate the prediction to the output tensor
            output = torch.cat((output, prediction_t), dim=1)

        return output
