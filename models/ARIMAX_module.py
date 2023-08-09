import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ARIMAX(nn.Module):
    def __init__(self, regressor_dim, sequence_length, ar_order, exogenous_dim):
        super(ARIMAX, self).__init__()

        self.regressor_dim = regressor_dim
        self.sequence_length = sequence_length
        self.ar_order = ar_order
        self.exogenous_dim = exogenous_dim
        # Create single linear layer
        self.linear = nn.Linear(self.regressor_dim, 1, bias=False)

    def forward(self, y, u):
        batch_size = y.size(0)
        device = y.device 
        
        # Initialize the output
        output = torch.zeros(batch_size, self.sequence_length+self.ar_order+1).to(y.device)

        # Start by filling in the initial conditions for the output
        output[:, :self.ar_order+1] = y

        for t in range(self.ar_order+1, self.sequence_length+self.ar_order+1):
            # Use the most recent output values and exogenous variables as the regressor
            #regressor = torch.cat((output[:, (t-self.ar_order):t], u[:,t-self.ar_order].unsqueeze(1)), dim=1)
            regressor = torch.cat((output[:, (t-self.ar_order):(t)] - output[:, (t-self.ar_order-1):(t-1)],
                                    u[:,(t-self.ar_order):(t-self.ar_order+self.exogenous_dim)]), dim=1)
            # Compute the prediction
            prediction_t = self.linear(regressor)
    
            # Store the prediction in the output tensor
            output[:, t] = output[:, t-1] + prediction_t.squeeze()

        return output


class MultiModelARIX(nn.Module):
    def __init__(self, num_clusters, regressor_dim, sequence_length, ar_order, exogenous_dim):
        super(MultiModelARIX, self).__init__()
        self.num_clusters = num_clusters
        self.regressor_dim = regressor_dim
        self.sequence_length = sequence_length
        self.ar_order = ar_order
        self.exogenous_dim = exogenous_dim
        # Create single linear layer with output size equal to num_clusters
        self.linear = nn.Linear(self.regressor_dim, self.num_clusters, bias=False)

    def forward(self, y, u):
        batch_size, num_clusters, _ = y.size()
    
        # Initialize the output with an extra cluster dimension
        output = torch.zeros(batch_size, num_clusters, self.sequence_length + self.ar_order + 1).to(y.device)
        output[:, :, :self.ar_order+1] = y

        for t in range(self.ar_order+1, self.sequence_length + self.ar_order + 1):
            # Use the most recent output values and exogenous variables as the regressor
            # Adjust the slicing to handle the extra cluster dimension
            regressor = torch.cat((output[:, :, (t-self.ar_order):(t)] - output[:, :, (t-self.ar_order-1):(t-1)],
                                u[:, :, (t-self.ar_order):(t-self.ar_order+self.exogenous_dim)]), dim=2)

            # Compute the prediction for all clusters simultaneously
            # Reshape the regressor to combine the batch and cluster dimensions
            regressor_combined = regressor.view(-1, self.regressor_dim)
            prediction_combined = self.linear(regressor_combined)
            
            # Reshape the prediction to separate the batch and cluster dimensions
            prediction_t = prediction_combined.view(batch_size, num_clusters)


            # Store the prediction in the output tensor
            output[:, :, t] = output[:, :, t-1] + prediction_t

        return output
