import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import copy
from sklearn.metrics import f1_score
from tqdm import tqdm


class SurvivalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(51984, 1200)
        self.fc2 = nn.Linear(1200, 340)
        self.fc3 = nn.Linear(340, 30)
        self.fc4 = nn.Linear(30, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_survivalnet(net, dataloaders, num_epochs = 20, metrics={'f1_score': f1_score}):
    running_loss = 0.0
    best_loss = 1e10

    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())

    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
            [f'Train_{m}' for m in metrics.keys()] + \
            [f'Test_{m}' for m in metrics.keys()]

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                net.train()
            else:
                net.eval()

            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].float()
                targets = sample['target'].float()
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = net(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = targets.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(metric(y_true.astype('uint8'), y_pred))

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
                # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(net.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    net.load_state_dict(best_model_wts)


def get_survival_dataloaders(images, target, batch_size=14):
    image_datasets = {
        x: Survival_Dataset(
            images,
            target,
            subset=x)
        for x in ['Train', 'Test']
    }

    dataloaders = {
            x: DataLoader(image_datasets[x],
                        batch_size=batch_size,
                        )
            for x in ['Train', 'Test']
        }
    return dataloaders


class Survival_Dataset(Dataset):
    def __init__(self,
                 images,
                 targets,
                 fraction = 0.1,
                 subset = None,
                 image_color_mode = "rgb"
                 ) -> None:

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode

        if subset not in ["Train", "Test"]:
            raise (ValueError(
                f"{subset} is not a valid input. Acceptable values are Train and Test."
            ))

        self.fraction = fraction
        self.images = images
        self.targets = targets

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        target = self.targets[index]
        
        sample = {"image": image, "target": target}
        sample["image"] = torch.tensor(sample["image"]).expand(3, -1, -1).type(torch.ShortTensor)
        sample["target"] = target
        return sample