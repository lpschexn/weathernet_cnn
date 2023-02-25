import  weathernet_cnn_lib as wn
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import WADSDataset
from stats import calc_true_positive_rate, calc_precision, calc_false_positive_rate

PATH = 'models/wn_0.pth'

# Set up model based on desired model
model = wn.LiLaNet(num_classes=2)
model.load_state_dict(torch.load(PATH))

# Set up data loader
dataset = WADSDataset('res/test')
data_loader = DataLoader(dataset, batch_size=1, shuffle=None)

# Testing Loop
with torch.no_grad():
    for i, data in enumerate(data_loader, 0):
        # Get inputs and labels
        inputs, targets = data

        outputs = model(inputs)

        # Print precision, recall, and false positive rates
        precision = calc_precision(outputs, targets)
        recall = calc_true_positive_rate(outputs, targets)
        false_positives = calc_false_positive_rate(outputs, targets)

        print(f'===============================================')
        print(f'Batch Number:                 {i}')
        print(f'Precision:                    {precision}')
        print(f'True Positive Rate (Recall):  {recall}')
        print(f'False Positive Rate:          {false_positives}')

print('*Oven Timer Ding* Finished Testing')