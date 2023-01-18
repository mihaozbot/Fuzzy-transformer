import numpy as np
from numpy import linalg as LA
import math

class Ellipse():
    def __init__(self, Sigma, mu, n_std = 2):
        self.Sigma = Sigma
        self.mu = mu
        self.n_std = n_std
        self.n_c = self.Sigma.shape[0]
        self.n_s = 100
        
    def confidence_ellipse(self):
        ellipse = np.zeros((self.n_c,self.n_s,2))
        w, v = LA.eig(self.Sigma)
        kot = np.linspace(0,2*math.pi,self.n_s,endpoint=True).reshape([-1,1])
        for index in range(self.mu.shape[0]):
            rotation = np.concatenate((np.cos(kot), np.sin(kot)), axis = 1)
            #print(np.diag(np.sqrt(w[index,:])))
            #print(v[index,:,:].shape)
            distance = np.matmul(np.diag(np.sqrt(w[index,:])),v[index,:,:])
            #print(distance.shape)
            ellipse[index,:] = np.matmul(rotation, distance)
            ellipse[index,:]  = ellipse[index,:]*self.n_std + self.mu[index,:]
           

        return ellipse