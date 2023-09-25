import torch
import os
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,transform):
        self.transform = transform
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.data_dir, file_name)
        image = Image.open(file_path)
        image = self.transform(image)
        # 在这里对图像进行一些转换
        return image
