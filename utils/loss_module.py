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
        sigma = torch.matmul((sigma_inv), (torch.transpose((sigma_inv), 2, 1)))
        d2_dS = torch.matmul(dl, sigma)
        dr = d.reshape(length, clusters, dim, 1)
        
        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters,1)
        #d2 = torch.matmul(dl, dr).reshape(length, clusters,1)
        psi = torch.nn.functional.softmax(-d2,0).reshape(length, 1, clusters)
        #loss_FCM = torch.sum(torch.mul(psi, d2).min(dim=2)[0])
        
        dist_FCM = torch.mul(torch.pow(psi,1), torch.matmul(dl, dr).reshape(length, 1, clusters))
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
        sigma_inv = torch.matmul((sigma_inv), (torch.transpose((sigma_inv), 2, 1)))
        sigma_calc = torch.mean(torch.matmul(dr, dl),dim=0)
        d2_calc = torch.matmul(torch.matmul(dl, torch.linalg.inv(sigma_calc)), dr).reshape(length, clusters,1)
        d2_dS = torch.matmul(dl, sigma_inv)
        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters,1)

        
        #loss_per_samle = torch.log(torch.det(sigma_calc))/2 + d2_calc.reshape(length, clusters)/2 - torch.log((1/clusters)*torch.ones(1,clusters).to(device))
        loss_per_samle = torch.log(torch.det(sigma_calc))/2 + d2_calc.reshape(length, clusters)/2 - torch.log((1/clusters)*torch.ones(1,clusters))
        loss = torch.sum(loss_per_samle)
        #(psi*torch.log(psi)).reshape(length, clusters))
        
        return loss

class GaussianMixtureLoss(torch.nn.Module):
    def __init__(self):
        super(GaussianMixtureLoss, self).__init__()
    
    def forward(self, z, mu, sigma_inv):
        length = z.shape[0]
        dim = z.shape[2]
        clusters = mu.shape[0]
        
        z = z.repeat(1, clusters, 1)
        mu = mu.reshape(1, clusters, dim)
        mu = mu.repeat(length, 1, 1)
        
        d = torch.sub(z, mu)
        dl = d.reshape(length, clusters, 1, dim)
        sigma_inv = torch.matmul((sigma_inv), (torch.transpose((sigma_inv), 2, 1)))
        d2_dS = torch.matmul(dl, sigma_inv)
        dr = d.reshape(length, clusters, dim, 1)
        
        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters,1)
        
        # Compute the negative log-likelihood of the data
        log_likelihood = -0.5 * (torch.log(torch.det(sigma_inv)) + d2.squeeze()) - dim * 0.5 * torch.log(2 * torch.tensor(3.141592653589793)).to(d.device)
        loss = -torch.mean(log_likelihood)
        
        return loss


#class GaussianMixtureLoss(torch.nn.Module):
#    def __init__(self):
#        super(GaussianMixtureLoss,self).__init__()
#    
#    def forward(self, z, mu, S_inv):
#         
#        length = z.shape[0]
#        dim = z.shape[2]
#        clusters = mu.shape[0]
#        z = z.repeat(1,clusters,1)
#        mu = mu.reshape(1,clusters,dim)
#        mu = mu.repeat(length,1,1)
#        d = torch.sub((mu),  z)
#        dl = d.reshape(-1,clusters, 1, dim)
#        Sigma_inv = torch.matmul((S_inv), (torch.transpose((S_inv), 2, 1)))
#        d2_dS = torch.matmul(dl, Sigma_inv)
#        dr = d.reshape(length, clusters, dim, 1)
#        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters,1)
#        psi = torch.nn.functional.softmax(-d2).reshape(length, 1, clusters)
#        
#        loss = torch.mean(torch.log(torch.det(torch.linalg.inv(Sigma_inv)))/2 + d2.reshape(length, clusters)/2)#to dela
#        #loss = -torch.mean(like)
#        ##(psi*torch.log(psi)).reshape(length, clusters))
#
#        return loss
    
class OverlappingLoss(torch.nn.Module):
    def __init__(self):
        super(OverlappingLoss,self).__init__()

    def forward(self, mu, sigma_inv):
        dim = mu.shape[1]
        clusters = mu.shape[0]
        
        mu = mu.reshape(1, clusters, dim)
        sigma = torch.inverse(torch.matmul((sigma_inv), (torch.transpose((sigma_inv), 2, 1))))
        d_B = torch.zeros(clusters,clusters)
        for i in range(clusters):
            for j in range(clusters):
                d_mu = mu[0,i]-mu[0,j]
                d_mu_t = d_mu.reshape(1, dim)
                sigma_ij = (sigma[i] + sigma[j])/2
                d_B[i,j] = (1/8)*torch.matmul(torch.matmul(d_mu_t,torch.inverse(sigma_ij)),d_mu) + (1/2)*torch.log(torch.det(sigma_ij)/torch.sqrt( torch.det(sigma[i])*torch.det(sigma[j]) ))
                if d_B[i,j] < 1e-3:
                    d_B[i,j] = 1e-3
                if d_B[i,j] > 2:
                    d_B[i,j] = 2

        loss_B = torch.sum(1/d_B)
        return loss_B
    
    
class FuzzyContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        super(FuzzyContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z, mu, sigma_inv):
        length = z.shape[0]
        dim = z.shape[2]
        clusters = mu.shape[0]

        z = z.repeat(1, clusters, 1)

        mu = mu.reshape(1, clusters, dim)
        mu = mu.repeat(length, 1, 1)

        d = torch.sub(z, mu)
        dl = d.reshape(length, clusters, 1, dim)
        sigma = torch.matmul(sigma_inv, torch.transpose(sigma_inv, 2, 1))
        d2_dS = torch.matmul(dl, sigma)
        dr = d.reshape(length, clusters, dim, 1)

        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters, 1)
        psi = torch.nn.functional.softmax(-d2, 0).reshape(length, 1, clusters)

        # Calculate loss for positive pairs
        pos_pairs = torch.mul(torch.pow(psi, 1), torch.matmul(dl, dr).reshape(length, 1, clusters))
        pos_loss = torch.sum(pos_pairs)

        # Calculate loss for negative pairs
        neg_pairs = torch.mul(torch.pow(1 - psi, 1), torch.clamp(self.margin - d2, min=0))
        neg_loss = torch.sum(neg_pairs)

        # Compute the overall loss
        loss_contrastive = (pos_loss + neg_loss) / length

        return loss_contrastive

    
class FuzzyCMeansLossWithPenalties(torch.nn.Module):
    def __init__(self):
        super(FuzzyCMeansLossWithPenalties, self).__init__()

    def forward(self, z, mu, sigma_inv, lambd1=1.0, lambd2=1.0, lambd3=1.0):
        length = z.shape[0]
        dim = z.shape[2]
        clusters = mu.shape[0]

        z = z.repeat(1, clusters, 1)

        mu = mu.reshape(1, clusters, dim)
        mu = mu.repeat(length, 1, 1)

        d = torch.sub(z, mu)
        dl = d.reshape(length, clusters, 1, dim)
        sigma = torch.matmul((sigma_inv), torch.transpose((sigma_inv), 2, 1))
        d2_dS = torch.matmul(dl, sigma)
        dr = d.reshape(length, clusters, dim, 1)

        d2 = torch.matmul(d2_dS, dr).reshape(length, clusters, 1)
        psi = torch.nn.functional.softmax(-d2, 1).reshape(length, 1, clusters)

        dist_FCM = torch.mul(torch.pow(psi, 1), torch.matmul(dl, dr).reshape(length, 1, clusters))
        loss_FCM = torch.sum(dist_FCM)

        dist_KM = torch.norm(z - mu, p=2, dim=2)
        loss_KM = dist_KM.min(dim=1)[0].sum()

        # Encourage equal spreading of samples among clusters
        equal_spread_penalty = torch.mean(torch.std(psi, dim=0))
        equal_spread_loss = lambd1 * equal_spread_penalty

        # Encourage samples to be pulled towards the clusters
        pull_towards_clusters_loss = lambd2 * torch.norm(torch.sub(z, mu), p=2)

        # Encourage smaller column norms for the sigma matrices
        column_norms = torch.norm((sigma_inv), dim=2)
        column_norm_penalty = torch.mean(column_norms)
        column_norm_loss = lambd3 * column_norm_penalty

        total_loss = loss_FCM + loss_KM + equal_spread_loss + pull_towards_clusters_loss + column_norm_loss

        return total_loss, loss_FCM, loss_KM, equal_spread_loss, pull_towards_clusters_loss,column_norm_loss

import torch
import torch.nn as nn

class TotalVariationLoss(torch.nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x, y):
        # Calculate the smoothness losses for x and y
        diff_x = torch.abs(x[:, :, :-1] - x[:, :, 1:])
        diff_y = torch.abs(y[:, :, :-1] - y[:, :, 1:])
        smoothness_loss_x = torch.mean(diff_x)
        smoothness_loss_y = torch.mean(diff_y)

        # Combine the losses by summing them
        combined_loss = smoothness_loss_x + smoothness_loss_y

        return combined_loss

class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, signals):
        loss = 0
        batch_size, num_models, signal_length = signals.size()

        for i in range(num_models):
            # Compute the mean signal for the batch and model i
            mean_signal = signals[:, i, :].mean(dim=0)

            for j in range(i + 1, num_models):
                # Compute the mean signal for the batch and model j
                other_signal = signals[:, j, :].mean(dim=0)

                # Compute the dot product between the mean signals and take the absolute value
                dot_product = torch.abs(torch.dot(mean_signal, other_signal))

                # Accumulate the dot product to the loss
                loss += dot_product

        return loss

class EntropyLoss(nn.Module):
    def __init__(self, lambda_entropy=0.1):
        super(EntropyLoss, self).__init__()
        self.lambda_entropy = lambda_entropy
        self.sm = torch.nn.Softmax(dim = 1)
    
    def compute_psi(self, mu, sigma_inv, z):
        d = torch.sub(mu, z)
        dl = d.reshape(z.shape[0], mu.shape[0], 1, mu.shape[1])
        
        sigma_inv_mat = torch.matmul(sigma_inv, torch.transpose(sigma_inv, 2, 1))
        d2_dS = torch.matmul(dl, sigma_inv_mat)

        dr = d.reshape(z.shape[0], mu.shape[0], mu.shape[1], 1)
        d2 = torch.clamp(torch.matmul(d2_dS, dr), min=1e-8)

        probs = self.sm(-d2.reshape(z.shape[0], mu.shape[0],1))
        psi = probs.reshape(z.shape[0], 1, mu.shape[0])
        return psi

    def forward(self, mu, sigma_inv, z):
        psi = self.compute_psi(mu, sigma_inv, z)

        # Entropy regularization term
        entropy = -1/torch.sum(psi * torch.log(psi + 1e-32), dim=-1)
        entropy_reg = self.lambda_entropy * entropy.mean()

        return entropy_reg
