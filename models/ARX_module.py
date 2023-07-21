import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ARX(nn.Module):
    def __init__(self, regressor_dim, sequence_length, ar_order):
        super(ARX, self).__init__()

        self.regressor_dim = regressor_dim
        self.sequence_length = sequence_length
        self.ar_order = ar_order
        # Create single linear layer
        self.linear = nn.Linear(self.regressor_dim, 1, bias=False)

    def forward(self, y, u):
        batch_size = y.size(0)
        device = y.device 
        
        # Initialize the output
        output = torch.zeros(batch_size, self.sequence_length+self.ar_order).to(y.device)

        # Start by filling in the initial conditions for the output
        output[:, :self.ar_order] = y

        for t in range(self.ar_order, self.sequence_length+self.ar_order):
            # Use the most recent output values and exogenous variables as the regressor
            regressor = torch.cat((output[:, (t-self.ar_order):t], 0*u[:,t-self.ar_order].unsqueeze(1)), dim=1)

            # Compute the prediction
            prediction_t = self.linear(regressor)
    
            # Store the prediction in the output tensor
            output[:, t] = prediction_t.squeeze()

        return output
