# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST,CIFAR10
import tqdm
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR

import matplotlib.pyplot as plt

from torchvision.utils import make_grid

# %% [markdown]
# ### Time Step Encoding using Gaussian Fourier Projection

# %%
class GaussianFourierProjection(nn.Module):
    
    """Gaussian random features for encoding time steps."""
    
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        
        # Randomly sample weights (frequencies) during initialization.
        # These weights (frequencies) are fixed during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        # Cosine(2 pi freq x), Sine(2 pi freq x)
        
        # Project the input x using randomly sampled frequencies (weights) W.
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    
    """A fully connected layer that reshapes outputs to feature maps.
      Allow time repr to input additively from the side of a convolution layer.
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Define a fully connected layer with input_dim and output_dim.
        self.dense = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Pass input x through the fully connected layer and reshape the output to 4D tensor.
        return self.dense(x)[..., None, None]  # 2D to 4D 

# %% [markdown]
# ### Noise Function

# %%
# Set the device for computation.
device = 'cuda:0'

# Set the value for the parameter sigma.
sigma = 25

def marginal_prob_std(t, sigma):
    
    """Computing the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE."""
    
    # Convert time steps to a PyTorch tensor with the specified device.
    t = torch.tensor(t, device=device)
    
    # Compute and return the standard deviation of the marginal probability.
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    
    """Computing the diffusion coefficient of our SDE.
    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE."""
    
    # Compute and return the diffusion coefficient based on the provided time steps and sigma.
    return torch.tensor(sigma**t, device=device)

# Create partial functions with fixed sigma for marginal probability and diffusion coefficient.
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

# %% [markdown]
# ## Unconditional DDPM

# %% [markdown]
# ### Loss Function for Denoising

# %%
# Loss Function for unconditional denoising
def loss_fn_uncond(model, x, marginal_prob_std, eps=1e-5):

    # Sample time uniformly in the interval [eps, 1 - eps]
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    
    # Finding the noise standard deviation at the sampled time `t`
    std = marginal_prob_std(random_t)
    
    # Generate normally distributed noise
    z = torch.randn_like(x, dtype=torch.float32)
    
    # Perturb the input data using the sampled noise and computed standard deviation
    perturbed_x = x + std[:, None, None, None] * z
    
    # Obtain the model's prediction on the perturbed data at the sampled time
    score = model(perturbed_x, random_t)
    
    # Compute the loss as the mean squared difference between the predicted score and perturbed data
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    
    return loss

# %% [markdown]
# ### U-Net Architecture for MNIST

# %%
class UNet_mnist(nn.Module):

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    
        super().__init__()
        
        # Gaussian feature embedding layer
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Encoding layers
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)  # Input: 1 channel, Output: channels[0]
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])  # 4 groups for group normalization

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)  # Input: channels[0], Output: channels[1]
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])  # 32 groups for group normalization

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)  # Input: channels[1], Output: channels[2]
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])  # 32 groups for group normalization

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)  # Input: channels[2], Output: channels[3]
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])  # 32 groups for group normalization
        
        # Decoding layers
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)  # Input: channels[3], Output: channels[2]
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])  # 32 groups for group normalization
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)  # Input: channels[2], Output: channels[1]
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])  # 32 groups for group normalization
        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)  # Input: channels[1], Output: channels[0]
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])  # 32 groups for group normalization
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)  # Input: channels[0], Output: 1 channel
        
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
    def forward(self, x, t, y=None):
        
        embed = self.act(self.time_embed(t))  # Gaussian feature embedding
        
        # Encoder
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        
        # Decoder
        h = self.tconv4(h4) + self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        k = self.dense7(embed)
        h = self.tconv2(h + h2) + k
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)
        
        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


# %% [markdown]
# ### Sampler for Generating Images using Random Noise

# %%
# Euler Maruyama Sampler code
num_steps = 1000

def Euler_Maruyama_sampler_mnist(score_model,
              marginal_prob_std,
              diffusion_coeff,
              batch_size=64,
              x_shape=(1, 28, 28),
              num_steps=num_steps,
              device='cuda:0',
              eps=1e-3, y=None):
    
    # Initialize time and random initial samples
    t = torch.ones(batch_size, device=device)  # Set initial time
    init_x = torch.randn(batch_size, *x_shape, device=device) \
    * marginal_prob_std(t)[:, None, None, None]  # Generate random initial samples
    
    # Define time steps and step size
    time_steps = torch.linspace(1., eps, num_steps, device=device)  # Generate time steps
    step_size = time_steps[0] - time_steps[1]  # Calculate step size
    x = init_x
    
    # Perform Euler Maruyama sampling
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step  # Set current time step
            g = diffusion_coeff(batch_time_step)  # Compute diffusion coefficient at the current time step
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size  # Euler-Maruyama update
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)  # Update samples
            
    return mean_x

# %% [markdown]
# 
# ### Training Loop for MNIST

# %%
# Training on MNIST dataset for unconditional denoising

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Initialize a UNet model for unconditional denoising
score_model = torch.nn.DataParallel(UNet_mnist(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

# Number of epochs
epochs = 100
# Size of a mini-batch
batch_size = 256
# Learning rate
lr = 5e-4

# Load MNIST dataset and create a data loader
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize Adam optimizer for the model parameters
optimizer = Adam(score_model.parameters())
tqdm_epoch = trange(epochs)

# List to store the training loss for each epoch
loss_mnist = []

# Training loop
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    
    # Iterate through the data loader
    for x, y in tqdm(dataloader):
        x = x.to(device)
        # Compute the loss for unconditional denoising
        loss = loss_fn_uncond(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        
        # Print the averaged training loss so far
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training
        torch.save(score_model.state_dict(), 'ckpt.pth')
    
    # Store the average loss for the epoch
    loss_mnist.append(avg_loss)
    
    ## Sample generation
    with torch.no_grad():
    
        sample_batch_size = 64 
        num_steps = 500 
        sampler = Euler_Maruyama_sampler_mnist 
        
        # Generate samples using the specified sampler
        samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device=device,
                  y=None)
        
        # Clamp the generated samples to be within [0.0, 1.0]
        samples = samples.clamp(0.0, 1.0)
        %matplotlib inline
        # Create a grid of sample images for visualization
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        # Save the sample grid as an image
        save_image(sample_grid, f"./final/ddpm_MNIST{epoch}.png")

# %% [markdown]
# ### Sample Output Images

# %%
import matplotlib.pyplot as plt
from PIL import Image

# Specify the file paths of the four images
image_paths = [
    "ddpm_MNIST10.png",
    "ddpm_MNIST34.png",
    "ddpm_MNIST70.png",
    "ddpm_MNIST85.png",
]

# Create a figure with four subplots
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

# Loop through each image path and display it in a subplot
for i, path in enumerate(image_paths):
    # Open the image using PIL
    img = Image.open(path)
    
    # Display the image in the corresponding subplot
    axs[i].imshow(img)
    axs[i].axis('off')
    

# Adjust layout to prevent overlapping
axs[0].set_title(f"Epoch 10")
axs[1].set_title(f"Epoch 34")
axs[2].set_title(f"Epoch 70")
axs[3].set_title(f"Epoch 85")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Loss per epoch for MNIST

# %%
# Normalize the training loss by the number of training samples
loss_mnist = np.array(loss_mnist) / 57000

# Plot the normalized training loss
plt.plot(loss_mnist)
plt.xlabel("No of Epochs")
plt.ylabel("Loss")
plt.title("MNIST")
plt.show()

# %% [markdown]
# ### U-Net Architecture for FashionMNIST

# %%
#------------------------------------U-net Architecture for FashionMNIST---------------------------

class UNet_Fashionmnist(nn.Module):

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    
        super().__init__()
        
        # Gaussian feature embedding layer
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Encoding layers
        self.conv0 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False, padding='same')  # Input: 1 channel, Output: channels[0]
        self.dense0 = Dense(embed_dim, channels[0])  # Input: embed_dim, Output: channels[0]
        self.gnorm0 = nn.GroupNorm(4, num_channels=channels[0])  # 4 groups for group normalization
        
        self.conv1 = nn.Conv2d(channels[0], channels[0], 3, stride=1, bias=False)  # Input: channels[0], Output: channels[0]
        self.dense1 = Dense(embed_dim, channels[0])  # Input: embed_dim, Output: channels[0]
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])  # 4 groups for group normalization

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)  # Input: channels[0], Output: channels[1]
        self.dense2 = Dense(embed_dim, channels[1])  # Input: embed_dim, Output: channels[1]
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])  # 32 groups for group normalization

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)  # Input: channels[1], Output: channels[2]
        self.dense3 = Dense(embed_dim, channels[2])  # Input: embed_dim, Output: channels[2]
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])  # 32 groups for group normalization
        
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)  # Input: channels[2], Output: channels[3]
        self.dense4 = Dense(embed_dim, channels[3])  # Input: embed_dim, Output: channels[3]
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])  # 32 groups for group normalization
        
        
        # Decoding layers
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)  # Input: channels[3], Output: channels[2]
        self.dense5 = Dense(embed_dim, channels[2])  # Input: embed_dim, Output: channels[2]
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])  # 32 groups for group normalization
        
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)  # Input: channels[2], Output: channels[1]
        self.dense6 = Dense(embed_dim, channels[1])  # Input: embed_dim, Output: channels[1]
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])  # 32 groups for group normalization

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)  # Input: channels[1], Output: channels[0]
        self.dense7 = Dense(embed_dim, channels[0])  # Input: embed_dim, Output: channels[0]
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])  # 32 groups for group normalization

        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)  # Input: channels[0], Output: 1 channel
        
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
    def forward(self, x, t, y=None):
        
        embed = self.act(self.time_embed(t))  # Gaussian feature embedding
        
        # Encoder
        h0 = self.conv0(x) + self.dense0(embed)  # Input: x, Output: channels[0]
        h0 = self.act(self.gnorm0(h0))
        h1 = self.conv1(h0) + self.dense1(embed)  # Input: h0, Output: channels[0]
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)  # Input: h1, Output: channels[1]
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)  # Input: h2, Output: channels[2]
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)  # Input: h3, Output: channels[3]
        h4 = self.act(self.gnorm4(h4))
        
        # Decoder
        h = self.tconv4(h4) + self.dense5(embed)  # Input: h4, Output: channels[2]
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.dense6(embed)  # Input: h + h3, Output: channels[1]
        h = self.act(self.tgnorm3(h))
        k = self.dense7(embed)  # Input: embed_dim, Output: channels[0]
        h = self.tconv2(h + h2) + k  # Input: h + h2, Output: channels[0]
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)  # Input: h + h1, Output: 1 channel
        
        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

# %% [markdown]
# ### Training Loop for FashionMNIST

# %%
# Training on FashionMNIST dataset for unconditional denoising

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Initialize a UNet model for unconditional denoising on FashionMNIST
score_model = torch.nn.DataParallel(UNet_Fashionmnist(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

# Number of epochs
epochs = 50
# Size of a mini-batch
batch_size = 256
# Learning rate
lr = 5e-4

# Load FashionMNIST dataset and create a data loader
dataset = datasets.FashionMNIST('.', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize Adam optimizer for the model parameters
optimizer = Adam(score_model.parameters())
tqdm_epoch = trange(epochs)

# List to store the training loss for each epoch
loss_fashionmnist_ = []

# Training loop
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    
    # Iterate through the data loader
    for x, y in tqdm(dataloader):
        x = x.to(device)
        # Compute the loss for unconditional denoising
        loss = loss_fn_uncond(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        # Print the averaged training loss so far
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training
        torch.save(score_model.state_dict(), 'ckpt.pth')
    
    # Store the average loss for the epoch
    loss_fashionmnist_.append(avg_loss)
    
    ## Sample generation
    with torch.no_grad():
    
        sample_batch_size = 64 
        num_steps = 500 
        sampler = Euler_Maruyama_sampler_mnist 
        
        # Generate samples using the specified sampler
        samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device=device,
                  y=None)
        
        # Clamp the generated samples to be within [0.0, 1.0]
        samples = samples.clamp(0.0, 1.0)
        %matplotlib inline
        # Create a grid of sample images for visualization
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        # Save the sample grid as an image
        save_image(sample_grid, f"./final/ddpm_FashionMNIST{epoch}.png")

# %% [markdown]
# ### Sample Images

# %%
import matplotlib.pyplot as plt
from PIL import Image

# Specify the file paths of the four images
image_paths = [
    "ddpm_FashionMNIST2.png",
    "ddpm_FashionMNIST7.png",
    "ddpm_FashionMNIST18.png",
    "ddpm_FashionMNIST49.png",
]

# Create a figure with four subplots
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

# Loop through each image path and display it in a subplot
for i, path in enumerate(image_paths):
    # Open the image using PIL
    img = Image.open(path)
    
    # Display the image in the corresponding subplot
    axs[i].imshow(img)
    axs[i].axis('off')
    

# Adjust layout to prevent overlapping
axs[0].set_title(f"Epoch 2")
axs[1].set_title(f"Epoch 7")
axs[2].set_title(f"Epoch 18")
axs[3].set_title(f"Epoch 49")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Loss per Epoch for FashionMNIST

# %%
# Normalize the training loss by the number of training samples
loss_fashionmnist = np.array(loss_fashionmnist) / 57000

# Plot the normalized training loss
plt.plot(loss_fashionmnist)
plt.xlabel("No of Epochs")
plt.ylabel("Loss")
plt.title("FashionMNIST")
plt.show()

# %% [markdown]
# ### U-Net Architecture for CIFAR10

# %%
class UNet_cifar(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256,
               text_dim=256, nClass=10):

        """Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings of time.
          text_dim:  the embedding dimension of text / digits.
          nClass:    number of classes you want to model.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
            )
        # Encoding layers where the resolution decreases
        self.conv0 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False, padding='same')
        self.dense0 = Dense(embed_dim, channels[0])
        self.gnorm0 = nn.GroupNorm(4, num_channels=channels[0])  # Input: 3 channels, Output: channels[0]

        self.conv1 = nn.Conv2d(channels[0], channels[0], 3, stride=1, bias=False, padding='same')
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])  # Input: channels[0], Output: channels[0]

        self.conv2 = nn.Conv2d(channels[0], channels[1], 2, stride=2, bias=False)  # Input: channels[0], Output: channels[1]
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])  # Input: channels[1], Output: channels[1]

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=False, padding='same')  # Input: channels[1], Output: channels[2]
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])  # Input: channels[2], Output: channels[2]

        self.conv4 = nn.Conv2d(channels[2], channels[3], 2, stride=2, bias=False)  # Input: channels[2], Output: channels[3]
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])  # Input: channels[3], Output: channels[3]

        self.conv5 = nn.Conv2d(channels[3], channels[3], 2, stride=2, bias=False)  # Input: channels[3], Output: channels[3]
        self.dense5 = Dense(embed_dim, channels[3])
        self.gnorm5 = nn.GroupNorm(32, num_channels=channels[3])  # Input: channels[3], Output: channels[3]

        # Decoding layers where the resolution increases
        self.tconv5 = nn.ConvTranspose2d(channels[3], channels[3], 2, stride=2, bias=False)  # Input: channels[3], Output: channels[3]
        self.tdense5 = Dense(embed_dim, channels[3])
        self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])  # Input: channels[3], Output: channels[3]

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2, bias=False)  # Input: channels[3], Output: channels[2]
        self.tdense4 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])  # Input: channels[2], Output: channels[2]

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 1, stride=1, bias=False)  # Input: channels[2], Output: channels[1]
        self.tdense3 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])  # Input: channels[1], Output: channels[1]

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2, bias=False)  # Input: channels[1], Output: channels[0]
        self.tdense2 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])  # Input: channels[0], Output: channels[0]

        self.tconv1 = nn.ConvTranspose2d(channels[0], channels[0], 1, stride=1)  # Input: channels[0], Output: channels[0]
        self.tdense1 = Dense(embed_dim, channels[0])
        self.tgnorm1 = nn.GroupNorm(32, num_channels=channels[0])  # Input: channels[0], Output: channels[0]
        self.tconv0 = nn.ConvTranspose2d(channels[0], 3, 1, stride=1)  # Input: channels[0], Output: 3

        # The swish activation function
        self.act = nn.SiLU()  # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, y=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))
        # Encoding path
        h0 = self.conv0(x) + self.dense0(embed)  # Input: 3 channels, Output: channels[0]
        h0 = self.act(self.gnorm0(h0))
        h1 = self.conv1(h0) + self.dense1(embed)  # Input: channels[0], Output: channels[0]
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)  # Input: channels[0], Output: channels[1]
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)  # Input: channels[1], Output: channels[2]
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)  # Input: channels[2], Output: channels[3]
        h4 = self.act(self.gnorm4(h4))
        h5 = self.conv5(h4) + self.dense5(embed)  # Input: channels[3], Output: channels[3]
        h5 = self.act(self.gnorm5(h5))

        # Decoding path
        h = self.tconv5(h5) + self.tdense5(embed)  # Input: channels[3], Output: channels[3]
        h = self.act(self.tgnorm5(h))
        h = self.tconv4(h + h4) + self.tdense4(embed)  # Input: channels[3], Output: channels[2]
        ## Skip connection from the encoding path
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.tdense3(embed)  # Input: channels[2], Output: channels[1]
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2) + self.tdense2(embed)  # Input: channels[1], Output: channels[0]
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1) + self.tdense2(embed)  # Input: channels[0], Output: channels[0]
        h = self.act(self.tgnorm1(h))
        h = self.tconv0(h + h0)  # Input: channels[0], Output: 3

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


# %% [markdown]
# ### Sampler for generating images using model trained on CIFAR10 

# %%
# Euler Maruyama Sampler code for CIFAR
num_steps = 1000
def Euler_Maruyama_sampler_cifar(score_model,
              marginal_prob_std,
              diffusion_coeff,
              batch_size=64,
              x_shape=(3, 32, 32),
              num_steps=num_steps,
              device='cuda:0',
              eps=1e-3, y=None):
    
    # Initialize time and initial samples
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, *x_shape, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
    
    # Generate time steps
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    
    # Perform Euler-Maruyama sampling
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            
    return mean_x

# %% [markdown]
# ### Traning loop for CIFAR10

# %%
# Training on CIFAR10 dataset for unconditional denoising

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Initialize the model and move it to the specified device
score_model = torch.nn.DataParallel(UNet_cifar(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

# Number of epochs
epochs = 100
# Size of a mini-batch
batch_size = 256
# Learning rate
lr = 5e-4

# Load CIFAR10 dataset
dataset = datasets.CIFAR10('.', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize Adam optimizer for model parameters
optimizer = Adam(score_model.parameters())
tqdm_epoch = trange(epochs)

# List to store the training loss at each epoch
loss_cifar = []

for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    
    # Iterate through the dataloader for each mini-batch
    for x, y in tqdm(dataloader):
        x = x.to(device)
        # Calculate the unconditional denoising loss
        loss = loss_fn_uncond(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        
    # Printing the averaged training loss so far
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Updating the checkpoint after each epoch of training
    torch.save(score_model.state_dict(), 'ckpt.pth')
    
    # Append the average loss for the current epoch to the list
    loss_cifar.append(avg_loss)
    
    ## Sample generation
    with torch.no_grad():
    
        sample_batch_size = 64 
        num_steps = 500 
        sampler = Euler_Maruyama_sampler_cifar
        
        # Generate samples using the specified sampler
        samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device=device,
                  y=None)
        
        # Sample visualization
        samples = samples.clamp(0.0, 1.0)
        %matplotlib inline
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
        save_image(sample_grid, f"./final/ddpm_CIFAR{epoch}.png")


# %% [markdown]
# ### Sample Images

# %%
import matplotlib.pyplot as plt
from PIL import Image

# Specify the file paths of the four images
image_paths = [
    "_ddpm_sample_cifar43.png",
    "_ddpm_sample_cifar101.png",
]

# Create a figure with four subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Loop through each image path and display it in a subplot
for i, path in enumerate(image_paths):
    # Open the image using PIL
    img = Image.open(path)
    
    # Display the image in the corresponding subplot
    axs[i].imshow(img)
    axs[i].axis('off')
    

# Adjust layout to prevent overlapping
axs[0].set_title(f"Epoch 43")
axs[1].set_title(f"Epoch 100")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Loss per epoch for CIFAR10

# %%
# Multiply the loss values by the size of the dataset (number of training samples) for proper scaling
loss_cifar = np.array(loss_cifar) * 50000

# Plot the training loss over epochs
plt.plot(loss_cifar)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("CIFAR10")
plt.show()

# %% [markdown]
# # Conditional Diffusion

# %% [markdown]
# ### Word Embedding Layer

# %%
# Import necessary libraries
from einops import rearrange
import math

# Word Embeddings layers
class WordEmbed(nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super(WordEmbed, self).__init__()
        
        # Initialize an embedding layer with vocabulary size and embedding dimension
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)

    def forward(self, ids):
        # Forward pass through the embedding layer
        return self.embed(ids)

# %% [markdown]
# ### CrossAttention Network for Conditional DDPM

# %%
class CrossAttention(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=1,):
        """
        Note: For simplicity reason, we just implemented 1-head attention.
        """
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        
        # Linear layer for query
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        
        # Check if it's self-attention or cross-attention
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)     
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)  
            self.value = nn.Linear(context_dim, hidden_dim, bias=False) 
            
    def forward(self, tokens, context=None):
        # tokens: with shape [batch, sequence_len, hidden_dim]
        # context: with shape [batch, contex_seq_len, context_dim]
        if self.self_attn:
            Q = self.query(tokens)
            K = self.key(tokens)
            V = self.value(tokens)
        else:
            # implement Q, K, V for the Cross attention
            Q = self.query(tokens)
            K = self.key(context)
            V = self.value(context)

        # Inner product of K and Q
        scoremats = torch.einsum('bsh, bth -> bst', K, Q)

        # Compute the attention distribution using softmax
        attnmats = torch.softmax(scoremats / (Q.shape[-1] ** 0.5), dim=-1)

        # Weighted average value vectors by attnmats
        ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V)
        return ctx_vecs
    
class TransformerBlock(nn.Module):
   
    """The transformer block that combines self-attn, cross-attn and feed forward neural net"""
    def __init__(self, hidden_dim, context_dim):
        super(TransformerBlock, self).__init__()
        
        # Self-attention layer
        self.attn_self = CrossAttention(hidden_dim, hidden_dim)
        
        # Cross-attention layer
        self.attn_cross = CrossAttention(hidden_dim, hidden_dim, context_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward neural network
        hidden_units = 2 * hidden_dim
        self.ffn  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, hidden_dim)
        )

    def forward(self, x, context=None):
        # Notice the + x as residue connections
        x = self.attn_self(self.norm1(x)) + x
        # Notice the + x as residue connections
        x = self.attn_cross(self.norm2(x), context=context) + x
        # Notice the + x as residue connections
        x = self.ffn(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
   
    def __init__(self, hidden_dim, context_dim):
        super(SpatialTransformer, self).__init__()
        
        # Transformer block
        self.transformer = TransformerBlock(hidden_dim, context_dim)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        # Combine the spatial dimensions and move the channel dimen to the end
        x = rearrange(x, "b c h w->b (h w) c")
        # Apply the sequence transformer
        x = self.transformer(x, context)
        # Reverse the process
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # Residue
        return x + x_in

# %% [markdown]
# ### U-net Architecture for MNIST and FashionMNIST

# %%
class UNet_Transformer_mnist(nn.Module):
    
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256,
               text_dim=256, nClass=10):

        """Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings of time.
          text_dim:  the embedding dimension of text / digits.
          nClass:    number of classes you want to model.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
            )
        
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim)

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialTransformer(channels[3], text_dim)

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)     #  + channels[2]
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)     #  + channels[1]
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1) #  + channels[0]

        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(nClass, text_dim)

    def forward(self, x, t, y=None):
        
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)
        
        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h3 = self.attn3(h3, y_embed)  # Use your attention layers
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        h4 = self.attn4(h4, y_embed)

        # Decoding path
        h = self.tconv4(h4) + self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

# %% [markdown]
# ### Loss function for Conditional DDPM

# %%
def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-5):

    """Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    y: Labels for conditioning.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    # Generate random time steps between (eps, 1-eps)
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    # Generate random noise
    z = torch.randn_like(x)
    # Compute the standard deviation of the perturbation kernel
    std = marginal_prob_std(random_t)
    # Perturb the input data
    perturbed_x = x + z * std[:, None, None, None]
    # Model evaluation on perturbed data
    score = model(perturbed_x, random_t, y=y)
    # Compute the loss using the score and perturbation information
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss

# %% [markdown]
# ### Training loop for MNIST

# %%
score_model = torch.nn.DataParallel(UNet_Transformer_mnist(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

# Number of epochs
n_epochs = 100
# Size of a mini-batch
batch_size = 1024
# Learning rate
lr = 10e-4

# MNIST dataset loading
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# List to store conditional loss for each epoch
cond_loss_mnist = []

# Adam optimizer with a learning rate scheduler
optimizer = Adam(score_model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

# tqdm for epoch visualization
tqdm_epoch = trange(n_epochs)

# Training loop
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    
    # Iterate over the mini-batches
    for x, y in tqdm(data_loader):
        x = x.to(device)
        # Compute conditional loss
        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    
    # Learning rate scheduling
    scheduler.step()
    
    # Append the averaged conditional loss for the epoch
    cond_loss_mnist.append(avg_loss / num_items)
    lr_current = scheduler.get_last_lr()[0]
    
    # Print epoch summary
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    
    # Update the tqdm visualization
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    
    # Save the model checkpoint after each epoch
    torch.save(score_model.state_dict(), 'ckpt_transformer.pth')

# %% [markdown]
# ### Loss per epoch for MNIST

# %%
cond_loss_mnist = np.array(cond_loss_mnist)
plt.plot(cond_loss_mnist)
plt.xlabel("No of Epochs")
plt.ylabel("Conditional Loss")
plt.title("MNIST")
plt.show()

# %% [markdown]
# ### Sampler code for generation of Conditional MNIST Images

# %%
# Import necessary functions from torchvision
from torchvision.utils import save_image, make_grid

# Load the pre-trained checkpoint from disk.
device = 'cuda:0'

# Iterate over each digit (0 to 9)
for digit in range(0, 10):
    # Set the sample batch size and number of steps for the sampler
    sample_batch_size = 64
    num_steps = 500
    sampler = Euler_Maruyama_sampler_mnist

    # Generate samples using the specified sampler with the given digit label
    samples = sampler(score_model,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      num_steps=num_steps,
                      device=device,
                      y=digit * torch.ones(sample_batch_size, dtype=torch.long))

    # Clamp generated samples between 0.0 and 1.0
    samples = samples.clamp(0.0, 1.0)

    # Visualize the generated samples using matplotlib
    %matplotlib inline
    import matplotlib.pyplot as plt
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    # Plot the sample grid
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()

    # Save the sample grid as an image file
    save_image(sample_grid, f"./final/conditional_mnist_{digit}.png")

# %% [markdown]
# ### Training loop for FashionMNIST

# %%
# Import necessary modules from torchvision
from torchvision import datasets

# Initialize a DataParallel UNet_Transformer_mnist model with the specified marginal probability function
score_model = torch.nn.DataParallel(UNet_Transformer_mnist(marginal_prob_std=marginal_prob_std_fn))
# Move the model to the specified device (cuda:0)
score_model = score_model.to(device)

# Set the number of training epochs
n_epochs = 60
# Set the size of a mini-batch
batch_size = 256
# Set the learning rate
lr = 10e-4

# Load the FashionMNIST dataset with specified transformations
dataset = datasets.FashionMNIST('.', train=True, transform=transforms.ToTensor(), download=True)
# Create a DataLoader for the dataset
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# Initialize an empty list to store conditional losses during training
cond_loss_fashion = []

# Initialize an Adam optimizer for the model parameters with the specified learning rate
optimizer = Adam(score_model.parameters(), lr=lr)
# Initialize a learning rate scheduler using LambdaLR
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))
# Initialize a tqdm progress bar for the epochs
tqdm_epoch = trange(n_epochs)

# Training loop
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    
    # Iterate over batches in the data loader
    for x, y in tqdm(data_loader):
        # Move input data to the specified device
        x = x.to(device)
        # Calculate conditional loss
        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Accumulate loss and item count
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    
    # Update the learning rate using the scheduler
    scheduler.step()
    # Store the average conditional loss for the epoch
    cond_loss_fashion.append(avg_loss / num_items)
    # Get the current learning rate
    lr_current = scheduler.get_last_lr()[0]
    # Print epoch, average loss, and current learning rate
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    # Update the tqdm description with the averaged training loss
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Save the model checkpoint after each epoch of training
    torch.save(score_model.state_dict(), 'ckpt_transformer.pth')

# %% [markdown]
# ### Loss per epoch for FashionMNIST

# %%
cond_loss_fashion = np.array(cond_loss_fashion)
plt.plot(cond_loss_fashion)
plt.xlabel("No of Epochs")
plt.ylabel("Conditional Loss")
plt.title("FashionMNIST")
plt.show()

# %% [markdown]
# ### Sample code for Generation of Conditional FashionMNIST

# %%
# Import necessary modules from torchvision.utils
from torchvision.utils import save_image, make_grid

# Load the pre-trained checkpoint from disk.
device = 'cuda:0'

# Iterate over digits (0 to 9)
for digit in range(0, 10):
    # Convert digit to PyTorch tensor and long type
    digit = torch.Tensor([digit])
    digit = digit.long()

    # Set the sample batch size and number of steps
    sample_batch_size = 64
    num_steps = 500
    # Use Euler_Maruyama_sampler_mnist as the sampler
    sampler = Euler_Maruyama_sampler_mnist

    # Generate samples using the specified sampler.
    samples = sampler(score_model,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      num_steps=num_steps,
                      device=device,
                      y=digit)

    # Sample visualization.
    samples = samples.clamp(0.0, 1.0)
    
    # Display the sample grid using matplotlib
    %matplotlib inline
    import matplotlib.pyplot as plt
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()

    # Save the sample grid as an image file
    save_image(sample_grid, f"./final/conditional_fashionmnist_{digit}.png")

# %% [markdown]
# ### U-Net architecture for CIFAR10

# %%
# Define a time-dependent score-based model using U-Net architecture for CIFAR
class UNet_Transformer_cifar(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256,
                 text_dim=256, nClass=10):

        """Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings of time.
          text_dim:  the embedding dimension of text / digits.
          nClass:    number of classes you want to model.
        """
        super().__init__()

        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoding layers where the resolution decreases
        self.conv0 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False, padding='same')
        self.dense0 = Dense(embed_dim, channels[0])
        self.gnorm0 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv1 = nn.Conv2d(channels[0], channels[0], 3, stride=1, bias=False, padding='same')
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 2, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=False, padding='same')
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim)

        self.conv4 = nn.Conv2d(channels[2], channels[3], 2, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn4 = SpatialTransformer(channels[3], text_dim)

        self.conv5 = nn.Conv2d(channels[3], channels[3], 2, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[3])
        self.gnorm5 = nn.GroupNorm(32, num_channels=channels[3])
        self.attn5 = SpatialTransformer(channels[3], text_dim)

        # Decoding layers where the resolution increases
        self.tconv5 = nn.ConvTranspose2d(channels[3], channels[3], 2, stride=2, bias=False)
        self.tdense5 = Dense(embed_dim, channels[3])
        self.tgnorm5 = nn.GroupNorm(32, num_channels=channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2, bias=False)
        self.tdense4 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 1, stride=1, bias=False)  # + channels[2]
        self.tdense3 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2, bias=False)  # + channels[1]
        self.tdense2 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0], channels[0], 1, stride=1)  # + channels[0]
        self.tdense1 = Dense(embed_dim, channels[0])
        self.tgnorm1 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv0 = nn.ConvTranspose2d(channels[0], 3, 1, stride=1)

        # The swish activation function
        self.act = nn.SiLU()  # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(nClass, text_dim)

    def forward(self, x, t, y=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)

        # Encoding path
        h0 = self.conv0(x) + self.dense0(embed)
        h0 = self.act(self.gnorm0(h0))
        h1 = self.conv1(h0) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h3 = self.attn3(h3, y_embed)  # Use your attention layers
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        h4 = self.attn4(h4, y_embed)
        h5 = self.conv5(h4) + self.dense5(embed)
        h5 = self.act(self.gnorm5(h5))
        h5 = self.attn5(h5, y_embed)

        # Decoding path
        h = self.tconv5(h5) + self.tdense5(embed)
        h = self.act(self.tgnorm5(h))
        h = self.tconv4(h + h4) + self.tdense4(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.tdense3(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2) + self.tdense2(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1) + self.tdense2(embed)
        h = self.act(self.tgnorm1(h))
        h = self.tconv0(h + h0)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

# %% [markdown]
# ### Training loop for CIFAR10

# %%
# Create a DataParallel instance for the UNet_Transformer_cifar model
score_model = torch.nn.DataParallel(UNet_Transformer_cifar(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

# Import necessary scheduler modules
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

# Set the number of epochs, batch size, and learning rate
n_epochs = 60
batch_size = 256
lr = 0.0003

# Load CIFAR-10 dataset
dataset = CIFAR10('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize Adam optimizer with the model parameters and specified learning rate
optimizer = Adam(score_model.parameters(), lr=lr)

# Set up the learning rate scheduler
milestones = [5, 15, 25, 35, 60, 80, 100]
gamma = 0.7
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=0.001)
multistep = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
scheduler = cosine_scheduler

# Initialize tqdm progress bar for epochs
tqdm_epoch = trange(n_epochs)

# List to store conditional CIFAR-10 loss values
cond_cifar = []

# Main training loop
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0

    # Iterate through data loader batches
    for x, y in tqdm(data_loader):
        x = x.to(device)
        # Calculate conditional loss using the defined function
        loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update average loss and item count
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    # Adjust learning rate with the scheduler
    scheduler.step()

    # Append average loss to the list
    cond_cifar.append(avg_loss/num_items)

    # Adjust learning rate and scheduler at epoch 40
    if epoch == 40:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 3e-4
        scheduler = multistep

    # Fetch current learning rate
    lr_current = optimizer.param_groups[0]['lr']

    # Print epoch details
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))

    # Update the tqdm progress bar description
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

# %% [markdown]
# ### Loss per epoch for CIFAR10

# %%
cond_cifar = np.array(cond_cifar)
plt.plot(cond_cifar)
plt.xlabel("No of Epochs")
plt.ylabel("Conditional Loss")
plt.title("CIFAR10")
plt.show()

# %% [markdown]
# ### Sampler code for Conditional Generation of CIFAR10 Images

# %%
# Import necessary functions
from torchvision.utils import save_image, make_grid

# Specify the device for processing (CUDA or CPU)
device = 'cuda:0' 

# Iterate over digits 0 to 9
for digit in range(0, 10):
    # Convert digit to a PyTorch tensor and then to a long tensor
    digit = torch.Tensor([digit])
    digit = digit.long()

    # Set sample batch size and number of steps for the sampler
    sample_batch_size = 64 
    num_steps = 500 

    # Use Euler_Maruyama_sampler_cifar as the sampler
    sampler = Euler_Maruyama_sampler_cifar

    # Generate samples using the specified sampler
    samples = sampler(score_model,
                      marginal_prob_std_fn,
                      diffusion_coeff_fn,
                      sample_batch_size,
                      num_steps=num_steps,
                      device=device,
                      y=digit)

    # Clamp generated samples to the range [0.0, 1.0]
    samples = samples.clamp(0.0, 1.0)

    # Visualize the generated samples using matplotlib
    %matplotlib inline
    import matplotlib.pyplot as plt
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    # Display the sample grid
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()

    # Save the generated samples as an image
    save_image(sample_grid, f"./final/conditional_Cifarmnist_{digit}.png")

# %% [markdown]
# ## Stable Diffusion on MNIST and FashionMNIST

# %% [markdown]
# ### Autoencoder for Compressing Images to Latent Space 

# %%
import torch.nn as nn

class AutoEncoder(nn.Module):

    def __init__(self, channels=[4, 8, 32]):
        
        super().__init__()
        
        # Gaussian random feature embedding layer for time
        # Encoding layers where the resolution decreases
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[2])
        )
        
        # Decoding layers where the resolution increases
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=True),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=True, output_padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
            nn.ConvTranspose2d(channels[0], 1, 3, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through the encoder and decoder
        output = self.encoder(x)
        output = self.decoder(output)
        return output


# %%
x_tmp = torch.randn(1,1,28,28)
print(AutoEncoder()(x_tmp).shape)
assert AutoEncoder()(x_tmp).shape == x_tmp.shape, "Check conv layer spec! the autoencoder input output shape not align"

# %% [markdown]
# ### Loss function for AutoEncoder

# %%
from lpips import LPIPS
# Define the loss function, MSE and LPIPS
loss_fn_ae = lambda x,xhat: nn.functional.mse_loss(x, xhat) 

# %% [markdown]
# ### Training of AutoEncoder using FashionMNIST samples

# %%
from torchvision import datasets

# Create an instance of the AutoEncoder class with specific channel configurations
ae_model = AutoEncoder([4, 4, 4]).cuda()

# Number of training epochs
n_epochs = 50  

# Size of a mini-batch
batch_size = 256  

# Learning rate
lr = 1e-4 

# Specify the device for training (CUDA in this case)
device = "cuda:0"

# Load FashionMNIST dataset from torchvision
dataset = datasets.FashionMNIST('.', train=True, transform=transforms.ToTensor(), download=True)

# Create a DataLoader to handle batching, shuffling, and loading data in parallel
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# List to store autoencoder loss for each epoch
auto_loss = []

# Adam optimizer for updating autoencoder parameters
optimizer = Adam(ae_model.parameters(), lr=lr)

# tqdm_epoch provides a progress bar for training epochs
tqdm_epoch = trange(n_epochs)

# Loop through epochs
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    
    # Loop through batches in the DataLoader
    for x, y in data_loader:
        x = x.to(device)
        
        # Pass input through the encoder
        z = ae_model.encoder(x)
        
        # Pass the encoded representation through the decoder
        x_hat = ae_model.decoder(z)
        
        # Calculate the autoencoder loss
        loss = loss_fn_ae(x, x_hat)
        
        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update average loss and number of items processed
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    
    # Calculate and store the average loss for the epoch
    auto_loss.append(avg_loss / num_items)
    
    # Print the average loss for the epoch
    print('{} Average Loss: {:5f}'.format(epoch, avg_loss / num_items))
    
    # Update the progress bar description
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    
    # Save the checkpoint after each epoch of training
    torch.save(ae_model.state_dict(), 'ckpt_ae.pth')


# %% [markdown]
# ### Loss per epoch for AutoEncoder

# %%
auto_loss = np.array(auto_loss)
plt.plot(auto_loss)
plt.xlabel("No of Epochs")
plt.ylabel("Conditional Loss")
plt.title("AutoEncoder")
plt.show()

# %% [markdown]
# ### Visualization of AutoEncoder Input and Output

# %%
# Set the model to evaluation mode
ae_model.eval()

# Load a batch of data from the DataLoader
x, y = next(iter(data_loader))

# Pass the batch through the trained autoencoder and move the result to CPU
x_hat = ae_model(x.to(device)).cpu()

# Display the original images in a grid
plt.figure(figsize=(6, 6.5))
plt.axis('off')
plt.imshow(make_grid(x[:64, :, :, :].cpu()).permute([1, 2, 0]), vmin=0., vmax=1.)
plt.title("Original")
plt.show()

# Display the reconstructed images by the autoencoder in a grid
plt.figure(figsize=(6, 6.5))
plt.axis('off')
plt.imshow(make_grid(x_hat[:64, :, :, :].cpu()).permute([1, 2, 0]), vmin=0., vmax=1.)
plt.title("AE Reconstructed")
plt.show()


# %% [markdown]
# ### Creating the Latent Space Datasets for MNIST and FashionMNIST

# %%
# Set the batch size for data loading
batch_size = 256

# Create FashionMNIST and MNIST datasets with specified transformations
dataset1 = datasets.FashionMNIST('.', train=True, transform=transforms.ToTensor(), download=True)
dataset2 = datasets.MNIST('.', train=True, transform=transforms.ToTensor(), download=True)

# Create a DataLoader for MNIST dataset with specified batch size and other settings
data_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=4)

# Set the requires_grad attribute of the autoencoder model to False
ae_model.requires_grad_(False)

# Set the model to evaluation mode
ae_model.eval()

# Initialize empty lists to store encoded representations and corresponding labels
zs = []
ys = []

# Iterate over the DataLoader to encode images and collect labels
for x, y in tqdm(data_loader):
  # Encode the images using the autoencoder's encoder and move the result to CPU
  z = ae_model.encoder(x.to(device)).cpu()
  # Append the encoded representations and labels to the lists
  zs.append(z)
  ys.append(y)

# Concatenate the encoded representations and labels along the specified dimension
zdata = torch.cat(zs, )
ydata = torch.cat(ys, )

# %%
print(zdata.shape)
print(ydata.shape)
print(zdata.mean(), zdata.var())

# %%
from torch.utils.data import TensorDataset
latent_dataset = TensorDataset(zdata, ydata)

# %% [markdown]
# ### Latent U-Net Architecture for MNIST and FashionMNIST

# %%
# Define a class for the Latent UNet Transformer model
class Latent_UNet_Tranformer(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[4, 64, 128, 256], embed_dim=256,
                 text_dim=256, nClass=10):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[1])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[2])
        self.gnorm2 = nn.GroupNorm(4, num_channels=channels[2])
        self.attn2 = SpatialTransformer(channels[2], text_dim)
        self.conv3 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[3])
        self.gnorm3 = nn.GroupNorm(4, num_channels=channels[3])
        self.attn3 = SpatialTransformer(channels[3], text_dim)

        self.tconv3 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense6 = Dense(embed_dim, channels[2])
        self.tgnorm3 = nn.GroupNorm(4, num_channels=channels[2])
        self.attn6 = SpatialTransformer(channels[2], text_dim)
        self.tconv2 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)     # + channels[2]
        self.dense7 = Dense(embed_dim, channels[1])
        self.tgnorm2 = nn.GroupNorm(4, num_channels=channels[1])
        self.tconv1 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=1) # + channels[1]

        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(nClass, text_dim)

    def forward(self, x, t, y=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embed(t))
        y_embed = self.cond_embed(y).unsqueeze(1)
        # Encoding path
        ## Incorporate information from t
        h1 = self.conv1(x) + self.dense1(embed)
        ## Group normalization
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h2 = self.attn2(h2, y_embed)
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h3 = self.attn3(h3, y_embed)

        # Decoding path
        ## Skip connection from the encoding path
        h = self.tconv3(h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.attn6(h, y_embed)
        h = self.tconv2(h + h2)
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


# %% [markdown]
# ### Training Latent Diffusion Model

# %%
#Training Latent diffusion model

# Define a DataParallel wrapper for the Latent UNet Transformer model
latent_score_model = torch.nn.DataParallel(
    Latent_UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn,
                           channels=[4, 16, 32, 64], )
)
# Move the model to the specified device
latent_score_model = latent_score_model.to(device)

# Set the number of training epochs
n_epochs = 100 
## size of a mini-batch
# Set the batch size
batch_size = 256
## learning rate
# Set the learning rate
lr = 1e-4 

# Create a data loader for the latent dataset
latent_data_loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=True, )
# Set the model to training mode
latent_score_model.train()

# Initialize Adam optimizer for the model parameters
optimizer = Adam(latent_score_model.parameters(), lr=lr)
# Create a learning rate scheduler using LambdaLR
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.5, 0.995 ** epoch))
# Initialize tqdm for visualizing training progress
tqdm_epoch = trange(n_epochs)
# List to store the training loss for each epoch
latent_loss_mnist = []

# Training loop
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    # Iterate over batches in the data loader
    for z, y in latent_data_loader:
        # Move the latent variable z to the specified device
        z = z.to(device)
        # Compute the conditional loss using the latent score model
        loss = loss_fn_cond(latent_score_model, z, y, marginal_prob_std_fn)
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Update the model parameters
        optimizer.step()
        avg_loss += loss.item() * z.shape[0]
        num_items += z.shape[0]
    # Adjust the learning rate using the scheduler
    scheduler.step()
    # Append the averaged loss for the epoch to the list
    latent_loss_mnist.append(avg_loss/num_items)
    lr_current = scheduler.get_last_lr()[0]
    # Print the epoch, averaged loss, and current learning rate
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    # Update the tqdm description with the averaged loss
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Save the model checkpoint after each epoch of training
    torch.save(latent_score_model.state_dict(), 'ckpt_latent_diff_transformer.pth')


# %% [markdown]
# ### Loss per epoch for Stable Diffusion on FashionMNIST

# %%
latent_loss_mnist = np.array(latent_loss_mnist)
plt.plot(latent_loss_mnist)
plt.xlabel("No of Epochs")
plt.ylabel("Conditional Loss")
plt.title("Stable Diffusion on MNIST")
plt.show()

# %% [markdown]
# ## Sampler Code for Generation of Images using Stable Diffusion on FashionMNIST

# %%
sample_batch_size = 64 
num_steps = 500
# Set the sampler to Euler-Maruyama for MNIST
sampler = Euler_Maruyama_sampler_mnist
# Set the latent score model to evaluation mode
latent_score_model.eval()
# Specify the digit for conditional generation
digit = torch.Tensor([0])
digit = digit.long()

## Generate samples using the specified sampler.
# Use the latent score model to generate samples with the specified configuration
samples_z = sampler(latent_score_model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        num_steps=num_steps,
        device=device,
        x_shape=(4,10,10),
        y=digit)

## Sample visualization.
# Decode the generated samples using the autoencoder's decoder
decoder_samples = ae_model.decoder(samples_z).clamp(0.0, 1.0)
# Create a grid of the decoded samples for visualization
sample_grid = make_grid(decoder_samples, nrow=int(np.sqrt(sample_batch_size)))
# Display the sample grid
%matplotlib inline
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()

# Save the sample grid as an image file
from torchvision.utils import save_image, make_grid
save_image(sample_grid, f"./final/stable_dif_MNIST{0}.png")



