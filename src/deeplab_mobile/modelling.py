import numpy as np
from torchvision import models
import time
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score
import csv
import torch
from pathlib import Path
import os

from src.utils import TYPE_NAMES


def select_images_input(images_list, type_names=TYPE_NAMES, type_to_use = 'flair'):
    images_chosen = np.array([images_list[type_names.index(type_to_use)][i] for i in range(len(images_list[type_names.index(type_to_use)]))])
    images_seg = np.array([images_list[type_names.index('seg')][i] for i in range(len(images_list[type_names.index('seg')]))])
    return images_chosen, images_seg


def get_deeplab_mobile_model(pretrained=False, progress=True, num_classes = 3):
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        pretrained=pretrained,
        progress=progress,
        num_classes = num_classes)
    return model


def train_deeplab_mobile(model,
    dataloaders,
    metrics={'f1_score': f1_score}, 
    bpath=Path(os.getcwd()) / 'models',
    criterion = torch.nn.MSELoss(reduction='mean'), #TODO: torch.nn.CrossEntropyLoss(reduction='mean') gives an error but find another one that is good
    num_epochs = 2,
    ):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]

    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
            # Each epoch has a training and validation phase
            # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].float()
                masks = sample['mask'].float()
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
            print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

