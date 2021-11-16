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

    def __init__(self, df, path):
        # df is of type pd.DataFrame
        self.table = df
        self.path = path

    def __getitem__(self, index):
        image_arr = (self.table.iloc[index].to_numpy())
        image_name = image_arr[0]
        #print(image_name)

        image = Image.open(self.path+'cassava-leaf-disease-classification/train_images/'+image_name)
        
        #sqrWidth = np.ceil(np.sqrt(image.size[0]*image.size[1])).astype(int)
        
        image = image.resize((224, 224))
        
        #print(image)
        
        transformation = transforms.ToTensor()
        image = transformation(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #print(image.shape)
        #plt.imshow(image.permute(1, 2, 0))
        #plt.show()
        #print(image)
        
        image = torch.reshape(image, (3, 224, 224))
        image = normalize(image)
        return image, image_arr[1]

    def __len__(self):
        return len(self.table)
