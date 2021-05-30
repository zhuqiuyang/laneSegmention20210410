import torch
from utils.image_process import LaneDataset, ImageAug, DeformAug, ScaleAug, CutOut, ToTensor
from torchvision import transforms

csv_file = 'LMDB/train_list.csv'
lmdb_dir = 'LMDB'
transform = transforms.Compose([ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5), ToTensor()])

my_dataset = LaneDataset(csv_file, lmdb_dir, transform)
data_dict = my_dataset[10]
print(data_dict['image'].shape, data_dict['mask'].shape)
