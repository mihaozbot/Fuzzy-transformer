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

class EvolvingSystem(nn.Module):
    def __init__(self, input_length, output_dim, output_length,  num_clusters, latent_dim, regressor_dim, exogenous_dim, order, min_clamp, max_clamp):
        super(EvolvingSystem, self).__init__()
        self.input_length = input_length
        self.output_dim = output_dim
        self.output_length = output_length
        self.cluster_dim = latent_dim
        self.num_clusters = num_clusters
        self.regressor_dim = regressor_dim
        self.exogenous_dim = exogenous_dim
        self.order = order
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.mu = torch.nn.Parameter(data = 0.1*2*(torch.rand(self.num_clusters, self.cluster_dim)-1/2), requires_grad=True)
        self.sigma_inv = nn.Parameter(torch.zeros(self.num_clusters, self.cluster_dim, self.cluster_dim), requires_grad=True)
        with torch.no_grad():
            self.sigma_inv.diagonal(dim1=-2, dim2=-1).fill_(100)
        
        #Create separate ARX layers for fc_con and fc_recon
        self.fc_con_layers = nn.ModuleList([
            ARX_module.ARX(self.regressor_dim, self.output_length, self.order, self.exogenous_dim)
            for _ in range(self.num_clusters)
        ])

        # Create separate ARIMAX layers for fc_con and fc_recon

        #self.ARIX = ARIMAX_module.MultiModelARIX(self.num_clusters, self.regressor_dim, self.output_length, self.order, self.exogenous_dim)

        
        '''
        self.fc_recon_layers = nn.ModuleList([
            ARX_module.ARX(self.input_length+1, self.input_length, input_length)
            for _ in range(self.num_clusters)
        ])
        
        
        # Create separate LSTMDecoder layers for fc_con and fc_recon
        self.fc_con_layers = nn.ModuleList([
            LSTM_decoder_module.LSTMDecoder(self.regressor_dim-1, self.output_length, self.order)
            for _ in range(self.num_clusters)
        ])
        '''
        self.sm = torch.nn.Softmax(dim = 1)
        #self.ones = nn.Parameter(torch.ones((self.batch_size, 1, 1)), requires_grad=False)
        
    def softmax(x):
        # Ensure numerical stability by subtracting the maximum value
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        x_exp = torch.exp(x - max_val)
        
        # Compute the softmax probabilities
        softmax_probs = x_exp / torch.sum(x_exp, dim=1, keepdim=True)
    
        return softmax_probs

        #self.evol_drop_layer = nn.Dropout(p=0.5)
    def add_new_rule(self, z):
        print('TODO')
        
    def compute_centers(self, z):
        psi = self.compute_psi(z)

        mu = torch.sum(torch.einsum('bmi, bmj->bji', psi, z),0)
        mu /= psi.sum(dim=0).clamp_min_(1e-12)
        
        return mu

    
    def compute_psi(self, z):
        
        d = torch.sub((self.mu), z)
        dl = d.reshape(-1, self.num_clusters, 1, self.cluster_dim)
        
        #TEST OK -> self.mu-self.x_ant[0][0], d[0], dl[0]
        sigma_inv = torch.matmul((self.sigma_inv), torch.transpose((self.sigma_inv), 2, 1))

        d2_dS = torch.matmul(dl, sigma_inv)

        dr = d.reshape(-1, self.num_clusters, self.cluster_dim, 1)

        #d2 = torch.matmul(d2_dS, dr)
        d2 = torch.clamp( torch.matmul(d2_dS, dr), min=self.min_clamp)
        #d2 = torch.pow(d2, torch.pow(self.etta, 2)
        psi = self.sm(-d2.reshape(-1, self.num_clusters,1))
        #psi = self.evol_drop_layer(psi)

        psi = psi.reshape(-1, 1, self.num_clusters)
        return psi

    def compute_psi_detached(self, z):
        
        mu = self.mu.detach()
        sigma_inv = self.sigma_inv.detach()

        d = torch.sub((mu), z)
        dl = d.reshape(-1, self.num_clusters, 1, self.cluster_dim)

        #TEST OK -> self.mu-self.x_ant[0][0], d[0], dl[0]
        sigma_inv = torch.matmul((sigma_inv), torch.transpose((sigma_inv), 2, 1))

        d2_dS = torch.matmul(dl, sigma_inv)

        dr = d.reshape(-1, self.num_clusters, self.cluster_dim, 1)

        #d2 = torch.matmul(d2_dS, dr)
        d2 = torch.clamp( torch.matmul(d2_dS, dr), min=self.min_clamp)
        #d2 = torch.pow(d2, torch.pow(self.etta, 2)
        psi = self.sm(-d2.reshape(-1, self.num_clusters,1))
        #psi = self.evol_drop_layer(psi)

        psi = psi.reshape(-1, 1, self.num_clusters)
        return psi

    def forward(self, y, z, u, member=None):
        device = z.device 
        self.psi = self.compute_psi(z)  # Move z to the same device as self.psi
        
        if self.training:
            # Randomly switch between fuzzy strategy and winner-takes-all strategy
                winner_indices = self.psi.argmax(dim=2)
                winner_mask = F.one_hot(winner_indices, self.num_clusters).float().to(device)
        else:
            # Use fuzzy strategy during evaluation
            winner_mask = self.psi

        #winner_indices = self.psi.argmax(dim=2)
        #winner_mask = F.one_hot(winner_indices, self.num_clusters).float().to(device)

        y_con_i = []
        #x_recon_i = []
        y_LLM_list = []
        #x_LLM_list = []
        y_LLM_all_list = []
        for i in range(self.num_clusters):
            fc_con_layer = self.fc_con_layers[i].to(device)
            #fc_recon_layer = self.fc_recon_layers[i].to(device)

            y_LLM = fc_con_layer(y[:,i,:], u[:,i,:])
            y_LLM_list.append(y_LLM[:, -self.output_length:])
            y_LLM_all_list.append(winner_mask[:, :, i]*y_LLM[:, (-self.output_length-self.order):])
            
            y_con_i.append(winner_mask[:, :, i]* y_LLM[:, -self.output_length:])

            #x_LLM = fc_recon_layer(torch.zeros(self.batch_size,self.input_length).to(device), torch.zeros(self.batch_size,self.input_length ).to(device))
            #x_LLM_list.append(x_LLM[:, self.order:])
            
            #x_recon_i.append(winner_mask[:, :, i]* x_LLM)


        y_con = torch.stack(y_con_i, dim=1).sum(dim=1).unsqueeze(1)

        self.y_LLM_all = torch.stack(y_LLM_all_list, dim=1).sum(dim=1).unsqueeze(1)
        self.y_LLM = torch.stack(y_LLM_list, dim=1)

        return y_con
