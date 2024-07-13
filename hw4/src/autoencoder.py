import torch
from tqdm.auto import tqdm

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        # super(Autoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 1952),
        #     nn.Linear(1952, 488),
        #     nn.Linear(488, 244),
        #     nn.Linear(244, 122),
        #     nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(122, 244),
        #     nn.Linear(244, 488),
        #     nn.Linear(488, 1952),
        #     nn.Linear(1952, input_dim),
        # )
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, X, epochs=10, batch_size=32):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader(torch.Tensor(X), batch_size=batch_size, shuffle=True)
        
        train_losses = []  # List to store the training losses
        
        for epoch in range(epochs):
            running_loss = 0.0
            for data in dataloader:
                optimizer.zero_grad()
                reconstructed_data = self(data)
                loss = criterion(reconstructed_data, data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss/len(dataloader))  # Append the average loss for the epoch
            
        # # Plotting the training curve
        # plt.plot(range(1, epochs+1), train_losses)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training Curve - Autoencoder')
        # plt.show()
    
    def transform(self, X):
        with torch.no_grad():
            encoded = self.encoder(torch.Tensor(X))
            return encoded
    
    def reconstruct(self, X):
        with torch.no_grad():
            return self(torch.Tensor(X)).numpy()


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim, encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = x + noise
        return noisy_x
    
    def fit(self, X, epochs=10, batch_size=32):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader(torch.Tensor(X), batch_size=batch_size, shuffle=True)
        
        train_losses = []  # List to store the training losses
        
        for epoch in range(epochs):
            running_loss = 0.0
            for data in dataloader:
                optimizer.zero_grad()
                noisy_data = self.add_noise(data)
                reconstructed_data = self(noisy_data)
                loss = criterion(reconstructed_data, data)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss/len(dataloader))  # Append the average loss for the epoch
            
        # # Plotting the training curve
        # plt.plot(range(1, epochs+1), train_losses)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Training Curve - DenoisingAutoencoder')
        # plt.show()
