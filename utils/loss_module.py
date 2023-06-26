import torch
from scipy.stats import multivariate_normal as mvn
class KMeansClusteringLoss(torch.nn.Module):
    def __init__(self):
        super(KMeansClusteringLoss, self).__init__()

    def forward(self, z, mu):
        length = z.shape[0]
        dim = z.shape[2]
        clusters = mu.shape[0]

        z = z.repeat(1,clusters,1)

        mu = mu.reshape(1, clusters, dim)
        mu = mu.repeat(length, 1, 1)

        dist_KM = (z-mu).norm(2, dim=2)
        
        loss_KM = torch.mean(dist_KM.min(dim=1)[0])
        
        return loss_KM
    
class FuzzyCMeansClusteringLoss(torch.nn.Module):
    def __init__(self):
        super(FuzzyCMeansClusteringLoss, self).__init__()

    def forward(self, z, mu, sigma_inv):
        length = z.shape[0]
        dim = z.shape[2]
        clusters = mu.shape[0]

        z = z.repeat(1,clusters,1)

        mu = mu.reshape(1, clusters, dim)
        mu = mu.repeat(length, 1, 1)

        d = torch.sub(z, mu)
        dl = d.reshape(length, clusters, 1, dim)
        sigma = torch.matmul(torch.exp(sigma_inv), (torch.transpose(torch.exp(sigma_inv), 2, 1)))
        d2_dS = torch.matmul(dl, sigma)
        dr = d.reshape(length, clusters, dim, 1)
        
        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters,1)
        #d2 = torch.matmul(dl, dr).reshape(length, clusters,1)
        psi = torch.nn.functional.softmax(-d2,0).reshape(length, 1, clusters)
        #loss_FCM = torch.sum(torch.mul(psi, d2).min(dim=2)[0])
        
        dist_FCM = torch.mul(torch.pow(psi,3), torch.matmul(dl, dr).reshape(length, 1, clusters))
        loss_FCM = torch.sum(dist_FCM)
        
        dist_KM = (z-mu).norm(2, dim=2)
        
        loss_KM = dist_KM.min(dim=1)[0].sum()
        
        return loss_FCM
    
class LocalityPreservingLoss(torch.nn.Module):
    def __init__(self):
        super(LocalityPreservingLoss,self).__init__()
    
    def forward(self, x, z):
         
        loss = 0
        return loss
    
        
class QuadraticDiscriminantAnalysis(torch.nn.Module):
    def __init__(self):
        super(QuadraticDiscriminantAnalysis,self).__init__()
    
    def forward(self, z, mu, sigma_inv):
         
        length = z.shape[0]
        dim = z.shape[2]
        clusters = mu.shape[0]
        z = z.repeat(1,clusters,1)
        mu = mu.reshape(1,clusters,dim)
        mu = mu.repeat(length,1,1)
        d = torch.sub((mu),  z)
        dl = d.reshape(-1,clusters, 1, dim)
        dr = d.reshape(length, clusters, dim, 1)
        sigma = torch.matmul(torch.exp(sigma_inv), (torch.transpose(torch.exp(sigma_inv), 2, 1)))
        sigma_calc = torch.mean(torch.matmul(dr, dl),dim=0)
        d2_calc = torch.matmul(torch.matmul(dl, torch.linalg.inv(sigma_calc)), dr).reshape(length, clusters,1)
        d2_dS = torch.matmul(dl, sigma)
        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters,1)

        
        #loss_per_samle = torch.log(torch.det(sigma_calc))/2 + d2_calc.reshape(length, clusters)/2 - torch.log((1/clusters)*torch.ones(1,clusters).to(device))
        loss_per_samle = torch.log(torch.det(sigma_calc))/2 + d2_calc.reshape(length, clusters)/2 - torch.log((1/clusters)*torch.ones(1,clusters))
        loss = torch.sum(loss_per_samle)
        #(psi*torch.log(psi)).reshape(length, clusters))
        
        return loss

class GaussianMixtureLoss(torch.nn.Module):
    def __init__(self):
        super(GaussianMixtureLoss,self).__init__()
    
    def forward(self, z, mu, S_inv):
         
        length = z.shape[0]
        dim = z.shape[2]
        clusters = mu.shape[0]
        z = z.repeat(1,clusters,1)
        mu = mu.reshape(1,clusters,dim)
        mu = mu.repeat(length,1,1)
        d = torch.sub((mu),  z)
        dl = d.reshape(-1,clusters, 1, dim)
        Sigma_inv = torch.matmul(torch.exp(S_inv), (torch.transpose(torch.exp(S_inv), 2, 1)))
        d2_dS = torch.matmul(dl, Sigma_inv)
        dr = d.reshape(length, clusters, dim, 1)
        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters,1)
        psi = torch.nn.functional.softmax(-d2).reshape(length, 1, clusters)
        
        loss = torch.mean(torch.log(torch.det(torch.linalg.inv(Sigma_inv)))/2 + d2.reshape(length, clusters)/2)#to dela
        #loss = -torch.mean(like)
        ##(psi*torch.log(psi)).reshape(length, clusters))

        return loss
    
class OverlappingLoss(torch.nn.Module):
    def __init__(self):
        super(OverlappingLoss,self).__init__()

    def forward(self, mu, sigma_inv):
        dim = mu.shape[1]
        clusters = mu.shape[0]
        
        mu = mu.reshape(1, clusters, dim)
        sigma = torch.matmul(torch.exp(sigma_inv), (torch.transpose(torch.exp(sigma_inv), 2, 1)))
        d_B = torch.zeros(clusters,clusters)
        for i in range(clusters):
            for j in range(clusters):
                d_mu = mu[0,i]-mu[0,j]
                d_mu_t = d_mu.reshape(1, dim)
                sigma_ij = (sigma[i] + sigma[j])/2
                d_B[i,j] = (1/8)*torch.matmul(torch.matmul(d_mu_t,torch.inverse(sigma_ij)),d_mu) + (1/2)*torch.log(torch.det(sigma_ij)/torch.sqrt( torch.det(sigma[i])*torch.det(sigma[j]) ))
                
        loss_B = -torch.mean(d_B)
        return loss_B