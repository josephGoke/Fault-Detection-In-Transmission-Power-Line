''' Variational Autoencoder Implementation for Fault Detection in Power Systems
This code defines a Variational Autoencoder (VAE) architecture using PyTorch. The VAE consists of an encoder that maps input data to a latent space, and a decoder that reconstructs the input from the latent representation. The VAE is trained to minimize the reconstruction loss and the Kullback-Leibler divergence between the learned latent distribution and a prior distribution (usually a standard normal distribution). This implementation can be used for anomaly detection in power systems by identifying deviations from normal operating conditions in the latent space.'''


import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def reparameterize(self, mu, logvar):
        # logvar = torch.clamp(logvar, min=-5, max=5)  # Prevent numerical instability
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence