from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import *
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config
from utils.n_adam import NAdam


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

device_list = [0]
train_net = 'deeplabv3p' # 'unet'
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}

def loss_func(predict, target, nbclasses, loss_type, epoch):
    ''' can modify or add losses '''
    if loss_type == 0:
        ce_loss = MySoftmaxCrossEntropyLoss(nbclasses=nbclasses)(predict, target.long())
    else:
        ce_loss = DiceLoss(nbclasses=nbclasses)(predict, target.long())
    return ce_loss


def train_epoch(net, epoch, dataLoader, optimizer, trainF, config):
    net.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        optimizer.zero_grad()
        out = net(image)
        mask_loss = loss_func(out, mask, config.NUM_CLASSES, config.LOSS_TYPE, epoch)
        total_mask_loss += mask_loss.item()
        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF, config):
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        out = net(image)
        mask_loss = loss_func(out, mask, config.NUM_CLASSES, config.LOSS_TYPE, epoch)
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    testF.write("Epoch:{} \n".format(epoch))
    miou = 0
    for i in range(8):
        iou_i = result["TP"][i]/result["TA"][i]
        result_string = "{}: {:.4f} \n".format(i, iou_i)
        print(result_string)
        testF.write(result_string)
        miou += iou_i
    miou /= 8
    miou_string = "{}: {:.4f} \n".format('miou', miou)
    print(miou_string)
    testF.write(miou_string)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()


def adjust_lr(optimizer, epoch):
    if epoch == 4:
        lr = 3e-4
    elif epoch == 6:
        lr = 5e-5
    elif epoch == 8:
        lr = 1e-5
    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    lane_config = Config()
    if os.path.exists(lane_config.SAVE_PATH):
        shutil.rmtree(lane_config.SAVE_PATH)
    os.makedirs(lane_config.SAVE_PATH, exist_ok=True)
    trainF = open(os.path.join(lane_config.SAVE_PATH, "train_log.csv"), 'w')
    testF = open(os.path.join(lane_config.SAVE_PATH, "val_log.csv"), 'w')
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset("data_list/train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
                                                                              ScaleAug(), CutOut(32, 0.5), ToTensor()]))

    train_data_batch = DataLoader(train_dataset, batch_size=2*len(device_list), shuffle=True, drop_last=True, **kwargs)
    val_dataset = LaneDataset("data_list/val.csv", transform=transforms.Compose([ToTensor()]))

    val_data_batch = DataLoader(val_dataset, batch_size=2*len(device_list), shuffle=False, drop_last=False, **kwargs)
    net = nets[train_net](lane_config)
    if torch.cuda.is_available():
        net = net.cuda(device=device_list[0])
        net = torch.nn.DataParallel(net, device_ids=device_list)
    if lane_config.OPTIMIZER == 0:
        optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR, momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)
    elif lane_config.OPTIMIZER == 1:
        optimizer = torch.optim.Adam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)
    elif lane_config.OPTIMIZER == 2:
        optimizer = NAdam(net.parameters(), lr=lane_config.BASE_LR, weight_decay=lane_config.WEIGHT_DECAY)
    for epoch in range(lane_config.EPOCHS):
        if lane_config.ADJUST_LR_ENABLE:
            adjust_lr(optimizer, epoch)
        train_epoch(net, epoch, train_data_batch, optimizer, trainF, lane_config)
        test(net, epoch, val_data_batch, testF, lane_config)
        torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
    trainF.close()
    testF.close()
    torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), lane_config.SAVE_PATH, "finalNet.pth.tar"))


if __name__ == "__main__":
    main()
