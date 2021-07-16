import torch.optim as optim
import torch.nn as nn
from encoder_decoder import Conv_Autoencoder


outputs = []
num_epochs = 3
weight_decay = 1e-5

model = Conv_Autoencoder()
criterion = nn.MSELoss()
lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-5)
