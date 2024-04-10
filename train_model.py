# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:13:47 2024

@author: jduij
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from optic_flow_dots_dataset import OpticFlowDotsDataset

class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5,5,5), padding=0, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=3)
        self.fc = nn.Linear(in_features=64*7*32*32, out_features=6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc(x)
        return x


def train_validate_test_model(dataset, num_epochs=100, batch_size=10, learning_rate=0.001):
    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device="cpu"
    
    model = Conv3DNet().to(device)  # Move the model to the device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Split dataset
    num_samples = len(dataset)
    train_size = int(0.70 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_targets in train_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)  # Move data to the device
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(batch_data.to(device)), batch_targets.to(device)).item() for batch_data, batch_targets in val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()*1000:.4f}, Val Loss: {val_loss*1000 / len(val_loader):.4f}")

    print("Training complete!")

    # Evaluate on test data
    test_loss = 0.0
    with torch.no_grad():
        for batch_data, batch_targets in test_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)  # Move data to the device
            outputs = model(batch_data)
            test_loss += criterion(outputs, batch_targets).item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    return model

# Example usage:
folder_path = "C:\\Users\\jduij\\Documents\\GitHub\\optic_flow_dots_data\\"
dataset = OpticFlowDotsDataset(folder_path)

trained_model = train_validate_test_model(dataset)
