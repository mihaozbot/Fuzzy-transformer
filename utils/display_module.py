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
import math
import os


def display_clustering(sigma_inv, mu, z):
    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    clusters = mu.shape[0]
    display.clear_output(wait=True) 

    sigma_inv = torch.matmul((sigma_inv), torch.transpose((sigma_inv), 2, 1))
    sigma = inv(sigma_inv.detach().cpu().numpy())
    nc_plot = mu.shape[0]
    sigma = sigma[0:nc_plot,0:2,0:2]
    mu = mu.detach().cpu().numpy()
    mu = mu[0:nc_plot,0:2]
    ellipse = ellipse_module.Ellipse(sigma,mu,1)
    ellipse_points = ellipse.compute_confidence_ellipse()
    ellipse_points = np.einsum('ijk->jik', ellipse_points)
    
    # Use a colormap
    color_map = plt.get_cmap('jet')  # 'rainbow' can be replaced with any other available colormap
    colors = [color_map(i) for i in np.linspace(0, 1, clusters)]
    
    for i, c in enumerate(colors):
        ax.plot(ellipse_points[:,i,0],ellipse_points[:,i,1], color=c)

    display.clear_output(wait=True)


def display_membership(psi, z, epoch, type):
        if not plt.get_fignums():
                print("No figure exists.")

        ax = plt.gca()
        num_clusters = psi.shape[2]
        for i in range(num_clusters):
                index = np.argmax(psi,2) == i
                #plt.plot(z[index,0], z[index,1],'.') #color=plt.cm.RdYlBu(i))
                ax.plot(z[index,0], z[index,1],'.', color = plt.gca().lines[i].get_color()) #color=plt.cm.RdYlBu(i))

        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

        plt.grid(False)
        
        save_dir = '../images/clusters'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, f'clusters_{epoch}_{type}.pdf')
        plt.savefig(save_path, format='pdf', transparent=True)
        plt.show()
        
        
def display_attention(att):
        im = plt.imshow(att)
        plt.colorbar(im)
        plt.show()

import matplotlib.pyplot as plt
import os
import math

def visualize_attention_weights(attention_weights, epoch, fill_figure=False):
    num_layers, num_heads, seq_length, _ = attention_weights.shape

    for layer_idx in range(num_layers):
        layer_weights = attention_weights[layer_idx]
        num_heads = layer_weights.size(0)

        # Calculate the number of rows and columns based on the number of heads
        num_cols = int(math.ceil(math.sqrt(num_heads)))
        num_rows = int(math.ceil(num_heads / num_cols))

        # Define a scale factor for the size of each subplot
        scale_factor = 2 

        # Calculate the figure size to maintain square subplots
        figsize = (num_cols * scale_factor, num_rows * scale_factor)

        plt.figure(figsize=figsize)

        for head_idx in range(num_heads):
            head_weights = layer_weights[head_idx]

            plt.subplot(num_rows, num_cols, head_idx + 1)
            plt.imshow(head_weights.detach().numpy(), cmap='hot', interpolation='nearest', aspect='auto')
            plt.axis('off')  # Remove the axis labels and ticks

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=1, hspace=1)
        plt.tight_layout(pad=0.1)

        save_dir = '../images/attention'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, f'attention_layer_{layer_idx + 1}_{epoch}.pdf')
        plt.savefig(save_path, format='pdf', transparent=True)
        plt.show()



'''      
def plot_llm(llm_data):
    # Move the tensor from GPU to CPU and detach the gradient
    llm_data = llm_data.detach().cpu().numpy()

    # Get the dimensions of the data
    batch_size, num_clusters, signal_length = llm_data.shape

    # Create a subfigure for each cluster
    fig, axs = plt.subplots(num_clusters, 1, figsize=(8, 4*num_clusters))

    # Plot the whole signal for all batches
    for cluster_idx in range(num_clusters):
        # Get the LLM data for the current cluster
        llm_cluster = llm_data[:, cluster_idx, :]

        # Plot the whole signal for each batch
        for batch_idx in range(batch_size):
            axs[cluster_idx].plot(np.arange(signal_length), llm_cluster[batch_idx])

        # Set the title and labels for the subfigure
        axs[cluster_idx].set_title(f'Cluster {cluster_idx + 1}')
        axs[cluster_idx].set_xlabel('Time step')
        axs[cluster_idx].set_ylabel('LLM Output')

    plt.tight_layout()
    plt.savefig('../images/LLMs.pdf')
    plt.show()
'''
# Calculate the figure size based on the number of subfigures

def visualize_inputs(u, epoch, type):
    # Move the tensor from GPU to CPU and detach the gradient
    u = u.detach().cpu().numpy()

    # Create a new figure
    plt.figure(figsize=(3,2))

    # For each batch, plot the signal
    plt.plot(u[0, :])

    # Add a title and labels
    plt.title(f'Input signals at epoch {epoch}, type {type}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Improve layout
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    
    # Define the directory to save the plot and create it if it doesn't exist
    save_dir = '../images/inputs'
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the full path for the output file
    save_path = os.path.join(save_dir, f'visualize_llm_square_{epoch}.pdf')


    # Save the figure
    plt.savefig(save_path, format='pdf', transparent=True)


def visualize_llm(llm_data,epoch):
        # Move the tensor from GPU to CPU and detach the gradient
        llm_data = llm_data.detach().cpu().numpy()

        # Get the dimensions of the data
        num_clusters, signal_length = llm_data.shape

        # Calculate the number of rows and columns for the subplots
        num_rows = int(np.ceil(np.sqrt(num_clusters)))
        num_cols = int(np.ceil(num_clusters / num_rows))

        # Calculate the figure size based on the number of subfigures
        figsize = (num_cols * 4, num_rows * 2)

        # Create the figure and subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)

        # Flatten the subplots array in case there is only one row or column
        axs = axs.flatten()

        # Plot the whole signal for all batches
        for cluster_idx in range(num_clusters):
                # Get the LLM data for the current cluster
                llm_cluster = llm_data[cluster_idx, :]

                # Plot the whole signal for each batch
                #for batch_idx in range(batch_size):
                axs[cluster_idx].plot(np.arange(signal_length), llm_cluster)

                # Set the title and labels for the subfigure
                axs[cluster_idx].set_title(f'Cluster {cluster_idx + 1}')
                axs[cluster_idx].set_xlabel('Time step')
                axs[cluster_idx].set_ylabel('LLM Output')

        
        # Remove any empty subplots
        for i in range(num_clusters, num_rows * num_cols):
                fig.delaxes(axs[i])

        plt.tight_layout()
        plt.grid(False)
        save_dir = '../images/models'
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, f'visualize_llm_square_{num_rows}x{num_cols}_{epoch}.pdf')
        plt.savefig(save_path, format='pdf', transparent=True)

        #plt.show()
