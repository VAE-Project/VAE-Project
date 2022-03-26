# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# Misc
import numpy as np
import pandas as pd
# My modules
from utils import to_device


# MedGan

class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.input_dim = args.input_dim
        self.embedding_dim = args.embedding_dim
        self.hidden = args.hidden_D
        self.device = args.device
        self.lambda_gp = args.lambda_gp

        self.input_layer = nn.Linear(self.input_dim, self.hidden[0])
        self.input_activation = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden)):
            self.layers.append(nn.Linear(self.hidden[i-1], self.hidden[i]))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(self.hidden[-1], 1)

        self.logs = {"train loss": [], "val loss": []}

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        Calculates the gradient penalty loss for WGAN GP
        https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
        """
        Tensor = torch.cuda.FloatTensor if self.device == "cuda" else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def loss(self, y_real, y_synthetic, gradient_penalty):
        return -torch.mean(y_real) + torch.mean(y_synthetic) + self.lambda_gp * gradient_penalty


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()

        self.random_dim = args.random_dim
        self.embedding_dim = args.embedding_dim
        self.hidden = args.hidden_G
        self.is_finetuning = args.is_finetuning
        if args.decoder is not None:
            self.decoder = args.decoder.to(args.device)
            if not self.is_finetuning:
                for params in self.decoder.parameters():
                    params.require_grad = False
        else:
            self.decoder = None

        self.input_layer = nn.Linear(self.random_dim, self.hidden[0])
        self.input_activation = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden)):
            self.layers.append(nn.Linear(self.hidden[i-1], self.hidden[i]))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(self.hidden[-1], self.embedding_dim)
        self.output_activation = nn.Tanh()

        self.logs = {"train loss": [], "val loss": []}

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_activation(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        if self.decoder is not None:
            x = self.decoder(x)
        return x

    def loss(self, y_synthetic):
        return -torch.mean(y_synthetic)


class GAN(nn.Module):

    def __init__(self, args):
        super(GAN, self).__init__()

        self.random_dim = args.random_dim
        self.embedding_dim = args.embedding_dim

        self.G = Generator(args)
        self.D = Discriminator(args)

        self.logs = {"approx. EM distance": []}


# Autoencoder

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden = args.hidden

        # Layers
        self.input_layer = nn.Linear(self.input_dim, self.hidden[0])
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden)):
            self.layers.append(nn.Linear(self.hidden[i-1], self.hidden[i]))
        self.output_layer = nn.Linear(self.hidden[-1], self.latent_dim)

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.output_layer(x)
        x = self.tanh(x)
        return x


class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden = args.hidden

        # Layers
        self.input_layer = nn.Linear(self.latent_dim, self.hidden[-1])
        self.layers = nn.ModuleList()
        for i in range(len(self.hidden)-1, 0, -1):
            self.layers.append(nn.Linear(self.hidden[i], self.hidden[i-1]))
        self.output_layer = nn.Linear(self.hidden[0], self.input_dim)
        
        # Activations
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.input_layer(x)
        x=self.relu(x)
        for layer in self.layers:
            x = layer(x)
            x=self.relu(x)
        x = self.output_layer(x)
        x=self.relu(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self, args):
        super(Autoencoder, self).__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden = args.hidden
        self.p_zero = args.p_zero

        self.device = args.device

        self.encoder = Encoder(args).to(self.device)
        self.decoder = Decoder(args).to(self.device)

        self.criterion = nn.MSELoss(reduction="sum")

        self.logs = {"train loss": [], "val loss": []}

    def forward(self, x):
        return self.decoder(self.encoder(x))


# modified place
class VAE(nn.Module):

    def __init__(self, args) -> None:
        super(VAE, self).__init__()

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.hidden = args.hidden
        self.logs = {"train loss": [], "val loss": []}

        # Encoder
        self.enc_input = nn.Linear(self.input_dim, self.hidden[0])
        self.enc_layers = nn.ModuleList()
        for i in range(1, len(self.hidden)):
            self.enc_layers.append(nn.Linear(self.hidden[i-1], self.hidden[i]))
        self.mu = nn.Linear(self.hidden[-1], self.latent_dim)
        self.logvar = nn.Linear(self.hidden[-1], self.latent_dim)

        # Decoder
        self.dec_input = nn.Linear(self.latent_dim, self.hidden[-1])
        self.dec_layers = nn.ModuleList()
        for i in range(len(self.hidden)-1, 0, -1):
            self.dec_layers.append(nn.Linear(self.hidden[i], self.hidden[i-1]))
        self.dec_output = nn.Linear(self.hidden[0], self.input_dim)

        # Activations
        self.tanh = nn.Tanh()

    def encode(self, x):
        x = self.enc_input(x)
        for layer in self.enc_layers:
            x = layer(x)
            x = self.tanh(x)
        return self.mu(x), self.logvar(x)

    def decode(self, z):
        x = self.dec_input(z)
        for layer in self.dec_layers:
            x = layer(x)
        return self.dec_output(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, reconstruction, mu, logvar):
        bce = F.binary_cross_entropy(reconstruction, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kl



# Gibbs Sampling

class Gibbs():

    def cond_ECDF(self, train: pd.DataFrame, sample: pd.Series, dimension: str):
        ecdf = train.copy()
        for col in train.columns.difference([dimension]):
            ecdf = ecdf[ecdf[col] == sample[col]]
        return ecdf

    def sample_dimension(self, ecdf: pd.DataFrame, dimension: str):
        return ecdf.sample(n=1, axis=0)[dimension].values[0]



