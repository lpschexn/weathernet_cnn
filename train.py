import  weathernet_cnn_lib as wn
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import WADSDataset
from stats import calc_true_positive_rate, calc_precision, calc_false_positive_rate

# Create WeatherNet model with goal of identifying falling snow
model = wn.LiLaNet(num_classes=2)

# Create a data loader
dataset = WADSDataset('res/train')
data_loader = DataLoader(dataset, batch_size=1, shuffle=None)

# Set up a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # Get inputs and labels
        inputs, targets = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Print stats
        running_loss += loss.item()
        if i % 5 == 4:

            # Print loss value
            print("=============================================")
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            running_loss = 0.0

            # Print precision, recall, and false positive rates
            precision = calc_precision(outputs, targets)
            recall = calc_true_positive_rate(outputs, targets)
            false_positives = calc_false_positive_rate(outputs, targets)

            print(f'Precision:                    {precision}')
            print(f'True Positive Rate (Recall):  {recall}')
            print(f'False Positive Rate:          {false_positives}')



print('*Oven Timer Ding* Finished Training')

PATH = './weathernet_trained.pth'
print('Saving model to:' + PATH)

torch.save(model.state_dict(), PATH)