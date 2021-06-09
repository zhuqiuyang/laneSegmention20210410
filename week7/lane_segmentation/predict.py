import os
import cv2
import torch
import numpy as np
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config
from utils.image_process import crop_resize_data
from utils.process_labels import decode_color_labels

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
# for dvi in range(torch.cuda.device_count()):
   # print(torch.cuda.get_device_name(dvi))

device_id = 0
predict_net = 'deeplabv3p'
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}

def load_model(model_path):

    lane_config = Config()
    net = nets[predict_net](lane_config)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda(device=device_id)
        map_location = 'cuda:%d' % device_id
    else:
        map_location = 'cpu'

    model_param = torch.load(model_path, map_location=map_location)['state_dict']
    model_param = {k.replace('module.', ''):v for k, v in model_param.items()}
    net.load_state_dict(model_param)
    return net


def img_transform(img):
    img = crop_resize_data(img)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...].astype(np.float32)
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img


def get_color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()
    pred = decode_color_labels(pred)
    pred = np.transpose(pred, (1, 2, 0))
    return pred 


def resize_padding(pred, image_size=(3384, 1020), offset=690):
    pred = cv2.resize(pred, image_size, interpolation=cv2.INTER_NEAREST)
    padding = np.zeros((offset, image_size[0], 3), dtype=np.uint8)
    pred = np.concatenate([padding, pred], axis=0)
    return pred


def main():
    test_dir = 'test_example'
    model_path = os.path.join(test_dir, predict_net + '_finalNet.pth.tar')
    print('Loading model...')
    net = load_model(model_path)
    net.eval()
    print('Finished.')

    img_path = os.path.join(test_dir, 'test.jpg')
    img = cv2.imread(img_path)
    img = img_transform(img)

    print('Model infering...')
    pred = net(img)
    color_mask = get_color_mask(pred)
    color_mask = resize_padding(color_mask)
    cv2.imwrite(os.path.join(test_dir, 'color_mask.jpg'), color_mask)
    print('Finished')


if __name__ == '__main__':
    main()

