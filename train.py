import  weathernet_cnn_lib as wn
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from dataset import WADSDataset
from stats import WNStatTracker

MODEL_SAVE_PATH = './weathernet_trained.pth'
TRAIN_DATA_PATH = 'CHANGEME'
INPUT_SIZE = (2, 64, 1024)

# Create WeatherNet model with goal of identifying falling snow
model = wn.LiLaNet(num_classes=2)
summary(model, INPUT_SIZE)

# Create a data loader
dataset = WADSDataset(TRAIN_DATA_PATH)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Set up a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set up stat tracker
stat_tracker = WNStatTracker()

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

            # Print precision, recall, and false positive rates
            precision, recall, false_positives = stat_tracker.calc_sample_stats(outputs, targets)

            # Print loss value and stats
            print("=============================================")
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
            running_loss = 0.0
            print(f'Precision:                    {precision}')
            print(f'True Positive Rate (Recall):  {recall}')
            print(f'False Positive Rate:          {false_positives}')



precision, recall, false_positives, total_tests = stat_tracker.calc_cumulative_stats()     

print('===============================================')
print('*Oven Timer Ding* Finished Training')
print(f'Cumulative Test Results')
print(f'Total Number of Tests:        {total_tests}')
print(f'Precision:                    {precision}')
print(f'True Positive Rate (Recall):  {recall}')
print(f'False Positive Rate:          {false_positives}')


print('Saving model to:' + MODEL_SAVE_PATH)

torch.save(model.state_dict(), MODEL_SAVE_PATH)