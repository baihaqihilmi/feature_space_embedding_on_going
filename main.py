from torchvision.models import vgg16 , VGG16_Weights
from dataset.skin_cancer import SkinCancerDataset
from torch.utils.data import Dataset , random_split
from torch.utils.data.dataloader import DataLoader 
import yaml
import torch
from torchvision import transforms
import os
from torch import nn
import torch.optim as optim
from tempfile import TemporaryDirectory
import time
import numpy as np
from torchsummary import summary

with open("config.yaml" , "r") as file :
    cfg = yaml.safe_load(file)

### Self Transform 

transform = transforms.Compose([transforms.Resize(224) , transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406] , [0.229, 0.224, 0.225]) 
])

##
ds = SkinCancerDataset(csv_file= cfg['data']['csv_dir'] , dir = cfg['data']['dir'] ,  transform = transform)

train_size = round(cfg["data"]["train_size_ratio"] * ds.__len__())
val_size = round(cfg["data"]["val_size_ratio"] * ds.__len__())
test_size =  round(cfg["data"]["test_size_ratio"] * ds.__len__())
train_dataset , val_dataset , test_dataset = random_split(ds , [train_size, val_size , test_size])
image_datasets = {"train" : train_dataset,
                  "test" : test_dataset,
                  "val":val_dataset}

dataloaders = {"train":  DataLoader(train_dataset , batch_size= 4 , drop_last= True ),
               "val" : DataLoader(val_dataset , batch_size= 4 , drop_last= True ),
               "test" : DataLoader(test_dataset , batch_size= 4 , drop_last= True )}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print(f"Train dataset  : {train_dataset.__len__()} with {len(dataloaders['train'])}")
print(f"test dataset  : {test_dataset.__len__()} with {len(dataloaders['test'])}")
print(f"val dataset  : {val_dataset.__len__()} with {len(dataloaders['val'])}")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ds.labels_map
n_class = len(ds.labels_map)

# ## Defining Model

model = vgg16(weights = VGG16_Weights.DEFAULT )
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, n_class))
model.to(device) 
## 
for param in model.features.parameters():  ## Freeze the feature extractor
    param.requires_grad = False


## Training time

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
NUM_EPOCHS = 24
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

if __name__ =="__main__":
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS)
    torch.save(model_ft , "best.pt")
    freeze_support()