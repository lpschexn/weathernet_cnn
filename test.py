import  weathernet_cnn_lib as wn
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import WADSDataset
from stats import WNStatTracker

PATH = 'models/wn_0.pth'

# Set up model based on desired model
model = wn.LiLaNet(num_classes=2)
model.load_state_dict(torch.load(PATH))

# Set up data loader
dataset = WADSDataset('res/test')
data_loader = DataLoader(dataset, batch_size=1, shuffle=None)

# Stat Tracker
stat_tracker = WNStatTracker()

total_tests = 0

# Testing Loop
with torch.no_grad():
    for i, data in enumerate(data_loader, 0):

        total_tests += 1

        # Get inputs and labels
        inputs, targets = data

        outputs = model(inputs)

        # Print precision, recall, and false positive rates
        precision = stat_tracker.calc_precision(outputs, targets)
        recall = stat_tracker.calc_true_positive_rate(outputs, targets)
        false_positives = stat_tracker.calc_false_positive_rate(outputs, targets)

        print(f'===============================================')
        print(f'Batch Number:                 {i}')
        print(f'Precision:                    {precision}')
        print(f'True Positive Rate (Recall):  {recall}')
        print(f'False Positive Rate:          {false_positives}')

print('===============================================')
print('*Oven Timer Ding* Finished Testing')

cumulative_precision       = stat_tracker.calc_cumulative_precision()
cumulative_recall          = stat_tracker.calc_cumulative_true_positive_rate()
cumulative_false_positives = stat_tracker.calc_cumulative_false_positive_rate()

print(f'Cumulative Test Results')
print(f'Total Number of Tests:        {total_tests}')
print(f'Precision:                    {cumulative_precision}')
print(f'True Positive Rate (Recall):  {cumulative_recall}')
print(f'False Positive Rate:          {cumulative_false_positives}')