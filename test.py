import  weathernet_cnn_lib as wn
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from semantic_kitti_data_handler import SemanticKittiDataHandler
from dataset import WADSDataset, convert_predicted_to_numpy
from stats import WNStatTracker
from roc_utils import Roc

SAVED_MODEL_PATH = 'weathernet_trained.pth'
TEST_DATA_PATH   = 'CHANGEME'

# Set up model based on desired model
model = wn.LiLaNet(num_classes=2)
model.load_state_dict(torch.load(SAVED_MODEL_PATH))

# Set up data loader
dataset = WADSDataset(TEST_DATA_PATH)
data_loader = DataLoader(dataset, batch_size=1, shuffle=None)

# Semantic kitti data handler
data_handler = SemanticKittiDataHandler(TEST_DATA_PATH)

# Stat Tracker
stat_tracker = WNStatTracker()

# Softmax
softmax = nn.Softmax2d()

probs_flat = torch.empty(1)
targets_flat = torch.empty(1)

# Testing Loop
with torch.no_grad():
    for i, data in enumerate(data_loader, 0):

        # Get inputs and labels
        inputs, targets = data

        outputs = model(inputs)

        probs = softmax(outputs)

        print(probs.shape)
        print(targets.shape)

        # Flatten arrays for ROC analysis
        probs_flat = torch.cat((probs_flat, torch.flatten(probs)))
        targets_flat = torch.cat((targets_flat, torch.flatten(targets)))
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

# ROC plotting
roc = Roc(targets_flat.numpy(), probs_flat.numpy())
roc.plot_roc()
roc.save('weathernet')