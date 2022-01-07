import torch
import PIL
import pandas as pd
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, df, path, is_train):
        # df is of type pd.DataFrame
        self.table = df
        self.path = path
        self.is_train = is_train

    def __getitem__(self, index):
        image_arr = (self.table.iloc[index].to_numpy())
        image_name = image_arr[0]
        #print(image_name)
        aug = image_arr[2]

        image = Image.open(self.path+'cassava-leaf-disease-classification/train_images/'+image_name)
        
        #sqrWidth = np.ceil(np.sqrt(image.size[0]*image.size[1])).astype(int)
        
        image = image.resize((448, 448))
        
        #print(image)
        
        transformation = transforms.ToTensor()
        image = transformation(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #print(image.shape)
        #plt.imshow(image.permute(1, 2, 0))
        #plt.show()
        #print(image)
        
        image = torch.reshape(image, (3, 448, 448))
        # pick the right transformation
        if self.is_train:
            if aug == 1:
                transformation = transforms.RandomHorizontalFlip(p=1)
            elif aug == 2:
                transformation = transforms.RandomVerticalFlip(p=1)
            elif aug == 3:
                transformation = transforms.GaussianBlur(3)
            elif aug == 4:
                transformation = transforms.RandomRotation((90, 90))
            # apply transformation if augmentation is required
            if aug != 0:
                image = transformation(image)

        image = normalize(image)
        return image, image_arr[1]

    def __len__(self):
        return len(self.table)
