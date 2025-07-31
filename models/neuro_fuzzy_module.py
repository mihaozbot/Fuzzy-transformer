from numpy.linalg import inv
import torch
import torch.nn as nn
import torch.nn.functional as F

from importlib import reload 
import models.ARX_module as ARX_module
reload(ARX_module)
import models.ARIMAX_module as ARIMAX_module
reload(ARIMAX_module)
import models.LSTM_decoder_module as LSTM_decoder_module
reload(LSTM_decoder_module)

class NeuroFuzzyLayer(nn.Module):
    def __init__(self, input_length, output_dim, output_length,  num_clusters, latent_dim, regressor_dim, order):
        super(NeuroFuzzyLayer, self).__init__()
        self.input_length = input_length
        self.output_dim = output_dim
        self.output_length = output_length
        self.cluster_dim = latent_dim
        self.num_clusters = num_clusters
        self.regressor_dim = regressor_dim
        self.order = order

        self.mu = torch.nn.Parameter(data = 0.1*2*(torch.rand(self.num_clusters, self.cluster_dim)-1/2), requires_grad=True)
        self.sigma_inv = nn.Parameter(torch.zeros(self.num_clusters, self.cluster_dim, self.cluster_dim), requires_grad=True)
        with torch.no_grad():
            self.sigma_inv.diagonal(dim1=-2, dim2=-1).fill_(100)

        self.lin = torch.nn.Linear(self.cluster_dim , self.num_clusters*output_length)

        self.sm = torch.nn.Softmax(dim = 3)

    def add_new_rule(self, z):                                
        print('TODO')
        
    def compute_centers(self, z):                                                                                                                                                                                                                                                                                
        psi = self.compute_psi(z)
        mu = torch.sum(torch.einsum('bmi, bmj->bji', psi, z),0)
        mu /= psi.sum(dim=0).clamp_min_(1e-12)
        
        return mu
    
    def compute_psi(self, z):
        
        d = torch.sub((self.mu), z.reshape(-1, self.input_length, 1, self.cluster_dim))
        dl = d.reshape(-1,self.input_length, self.num_clusters, 1, self.cluster_dim)
        dr = d.reshape(-1,self.input_length,  self.num_clusters, self.cluster_dim, 1)
        sigma_inv = torch.matmul((self.sigma_inv), torch.transpose((self.sigma_inv), 2, 1))
        d2 = torch.clamp( torch.matmul(torch.matmul(dl, sigma_inv), dr), min=1e-12)
        psi = self.sm(-d2.reshape(-1, self.input_length, 1, self.num_clusters))

        return psi  

    def forward(self, y, z):
        device = z.device 
        self.psi = self.compute_psi(z)  # Move z to the same device as self.psi
         
        if self.training:
            # Randomly switch between fuzzy strategy and winner-takes-all strategy
                winner_indices = self.psi.argmax(dim = 3)
                winner_mask = F.one_hot(winner_indices, self.num_clusters).float().to(device)
        else:
            # Use fuzzy strategy during evaluation
            winner_mask = self.psi
            
        out = torch.matmul(winner_mask, self.lin(y).reshape(-1, self.input_length,self.num_clusters, self.output_length))

        return out
