import  weathernet_cnn_lib as wn
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from semantic_kitti_data_loader import SemanticKittiDataLoader
from dataset import WADSDataset, convert_predicted_to_numpy
from stats import WNStatTracker

SAVED_MODEL_PATH = 'weathernet_trained.pth'
TEST_DATA_PATH   = 'CHANGEME'

# Set up model based on desired model
model = wn.LiLaNet(num_classes=2)
model.load_state_dict(torch.load(SAVED_MODEL_PATH))

# Set up data loader
dataset = WADSDataset(TEST_DATA_PATH)
data_loader = DataLoader(dataset, batch_size=1, shuffle=None)

# Semantic kitti data handler
data_handler = SemanticKittiDataLoader(TEST_DATA_PATH)

# Stat Tracker
stat_tracker = WNStatTracker()

# Testing Loop
with torch.no_grad():
    for i, data in enumerate(data_loader, 0):

        # Get inputs and labels
        inputs, targets = data

        outputs = model(inputs)

        semantic_prediction, instance_prediction = convert_predicted_to_numpy(outputs[0,:,:,:])
        data_handler.save_predicted(i, semantic_prediction, instance_prediction)

        # Generate precision, recall, and false positive stats
        precision, recall, false_positives = stat_tracker.calc_sample_stats(outputs, targets)

        print(f'===============================================')
        print(f'Batch Number:                 {i}')
        print(f'Precision:                    {precision}')
        print(f'True Positive Rate (Recall):  {recall}')
        print(f'False Positive Rate:          {false_positives}')


precision, recall, false_positives, total_tests = stat_tracker.calc_cumulative_stats()     

print('===============================================')
print('*Oven Timer Ding* Finished Testing')
print(f'Cumulative Test Results')
print(f'Total Number of Tests:        {total_tests}')
print(f'Precision:                    {precision}')
print(f'True Positive Rate (Recall):  {recall}')
print(f'False Positive Rate:          {false_positives}')