import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim, g_hidden_dim, g_output_dim):
        """
        Initializes the Generator class.
        It's used by the GAN to generate new samples, but also by the EDM.

        Args:
            latent_dim (int): The dimension of the latent space.
            g_hidden_dim (int): The dimension of the hidden layer in the generator.
            g_output_dim (int): The dimension of the output layer in the generator.
        """
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(latent_dim, g_hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        """
        Forward pass of the NN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.fc4(x)
    

class Discriminator(nn.Module):
    def __init__(self, d_input_dim, d_hidden_dim):
        """
        Initialize the discriminator class for GAN (also known as Critic).

        Args:
            d_input_dim (int): The input dimension of the discriminator.
            d_hidden_dim (int): The hidden dimension of the discriminator.
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, d_hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the discriminator.
        """
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)

        return torch.sigmoid(self.fc4(x))
