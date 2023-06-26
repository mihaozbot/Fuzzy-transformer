#from matplotlib.pyplot import cm
#from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from IPython import display
import torch
import numpy as np
from importlib import reload
from numpy.linalg import inv
import utils.ellipse_module as ellipse_module
reload(ellipse_module)

def display_clustering(sigma_inv, mu, z):
        #display.display(pl.gcf())
        clusters = mu.shape[0]
        display.clear_output(wait=True) 
        sigma_inv = torch.matmul(torch.exp(sigma_inv), torch.transpose(torch.exp(sigma_inv), 2, 1))
        sigma = inv(sigma_inv.detach().cpu().numpy())
        nc_plot = mu.shape[0]
        sigma = sigma[0:nc_plot,0:2,0:2]
        mu = mu.detach().cpu().numpy()
        mu = mu[0:nc_plot,0:2]
        ellipse = ellipse_module.Ellipse(sigma,mu,1)
        ellipse_points = ellipse.compute_confidence_ellipse()
        ellipse_points = np.einsum('ijk->jik', ellipse_points)
        plt.plot(ellipse_points[:,:,0],ellipse_points[:,:,1])
        #color = iter(cm.rainbow(np.linspace(0, 1, clusters)))
        #for i in range(clusters):
        #        c = next(color)
        #        plt.plot(ellipse_points[:,i,0],ellipse_points[:,i,1], c=c)
        
def display_membership(psi, z):
        #display.display(pl.gcf())   
        num_clusters = psi.shape[2]
        for i in range(num_clusters):
                index = np.argmax(psi,2) == i
                plt.plot(z[index,0], z[index,1],'.') #color=plt.cm.RdYlBu(i))
                #plt.plot(z[index,0], z[index,1],'.', color = plt.gca().lines[i+1].get_color()) #color=plt.cm.RdYlBu(i))
        plt.show()  
        
def display_attention(att):
        im = plt.imshow(att)
        plt.colorbar(im)
        plt.show()
