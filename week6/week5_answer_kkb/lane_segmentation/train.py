import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug, ScaleAug, CutOut, ToTensor

EPOCHS = 1

def main():
    train_dataset = LaneDataset("data_list/train.csv",
                        transform=transforms.Compose(
                            [ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5), ToTensor()]))
    val_dataset = LaneDataset("data_list/val.csv",
                        transform=transforms.Compose([ToTensor()]))

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False, **kwargs)

    for epoch in range(EPOCHS):
        for batch_item in train_dataloader:
            image, mask = batch_item['image'], batch_item['mask']
            print('train', image.shape, mask.shape)
        for batch_item in val_dataloader:
            image, mask = batch_item['image'], batch_item['mask']
            print('val', image.shape, mask.shape)

if __name__ == "__main__":
    main()
