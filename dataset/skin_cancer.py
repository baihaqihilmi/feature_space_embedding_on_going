from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from glob import glob
import torch
import yaml
from sklearn import preprocessing
import numpy as np

## THis dataset Can be Downloaded in## 

def one_hot_encode(label, num_classes):
    # Create a one-hot encoded tensor
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1.0
    return one_hot
## Since we only interested in Pixel values
class SkinCancerDataset(Dataset):
    def __init__(self , dir ,  csv_file , transform = False) :
        
        self.dir = dir
        self.images = sorted(glob(os.path.join(dir , "HAM10000_images_part_1/*.jpg" )) +  glob(os.path.join(dir , "HAM10000_images_part_2/*.jpg" )))
        self.transform = transform
        self.data = pd.read_csv(csv_file)   
        self.data = self.data.sort_values("image_id")
        self.label_fit = preprocessing.LabelEncoder().fit(self.data["dx_type"])
        self.labels_map = self.label_fit.classes_
        self.labels = self.label_fit.transform(self.data["dx_type"])
        
    def __getitem__(self, index) :
        image = Image.open(self.images[index]) 
        label = torch.as_tensor(np.array(self.labels[index]).astype('float')).type("torch.LongTensor") ## Need to be translated to Tensor
        if self.transform :
            image  = self.transform(image)

        return image , label

        
    def __len__(self) :
        return len(self.images)
    

# if __name__ == "__main__":
#     cfg = yaml.safe_load("../config.yaml")
#     ds = SkinCancerDataset(csv_file= "../" + cfg["data"]["csv_dir"] , dir = "../" + cfg["data"]["dir"] )
 
        