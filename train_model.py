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
import the_luggage as lgg
import math
import os
from datetime import datetime

class Conv3DNet(nn.Module):
    def __init__(self, input_shape:tuple, nfilters:int, filter_shape:tuple, n_out_features:int):
        # Check the inputs
        if len(input_shape)!=5 or not all(isinstance(x, int) and x > 0 for x in input_shape):
            raise ValueError("input_shape must be a 5-element tuple of positive ints")  
        if not type(nfilters)==int or nfilters<=0:
            raise ValueError("nfilters must be a positive int")
        if len(filter_shape)!=3 or not all(isinstance(x, int) and x > 0 for x in filter_shape):
            raise ValueError("filter_shape must be a 3-element tuple of positive ints")      
        # Get parameters in readable form
        # Specify maxpool3d kernel size
        
        # calculate the output shape of the conv3d and maxpool layers
        conv1_bcfhw=lgg.conv3d_output_shape(input_shape,nfilters,filter_shape);
        maxpool_fhw=(conv1_bcfhw[2],1,1);
        pool_bcfhw=lgg.maxpool3d_output_shape(conv1_bcfhw, maxpool_fhw, stride=(1,1,1))
        # 
        super(Conv3DNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_shape[1], out_channels=nfilters, kernel_size=filter_shape)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=maxpool_fhw)
        self.fc = nn.Linear(in_features=math.prod(pool_bcfhw[1:]), out_features=n_out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc(x)
        return x


def train_validate_test_model(dataset,
                              num_epochs:int=1, 
                              batch_size:int=20, 
                              learning_rate:float=0.001
                              ):
    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split dataset
    num_samples = len(dataset)
    train_size = int(0.70 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    

    # Get one batch from the trainloader, so we know the sizes of the input to the model and the target
    for one_batch_data, one_batch_targets in train_loader:
        break
    shape_in = tuple(one_batch_data.shape)
    n_out = len(one_batch_targets[0])
    
    # Initialize the neural network model
    model = Conv3DNet(input_shape = shape_in, 
                      nfilters = 64, 
                      filter_shape = (shape_in[2],5,5),
                      n_out_features = n_out
                      ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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

if __name__=="__main__":
    #folder_path = "C:\\Users\\jduij\\Documents\\GitHub\\optic_flow_dots_data\\"
    data_folder_path=os.path.dirname(__file__)+'_data'
    dataset = OpticFlowDotsDataset(data_folder_path)
    trained_model = train_validate_test_model(dataset,num_epochs=1,batch_size=20)
    model_folder_path=os.path.dirname(__file__)+'_models'
    model_filename=os.path.join(model_folder_path,'model_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.pth')
    torch.save(trained_model, model_filename)
