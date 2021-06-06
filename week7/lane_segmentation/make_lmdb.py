import cv2
import lmdb
import random
import numpy as np
from tqdm import tqdm

#data_list_fns = ['data_list/train.csv', 'data_list/val.csv']
data_list_fns = ['crop_imgs/img_list.csv']

keys = []
for data_list_fn in data_list_fns:
    with open(data_list_fn, 'r') as f:
        lines = f.readlines()  
        keys += [k for line in lines for k in line.strip().split(', ')]
        
#env = lmdb.open('small_dataset_lmdb', map_size=int(1e9))
env = lmdb.open('crop_imgs_lmdb', map_size=int(1e9))
txn = env.begin(write=True)
for key in tqdm(keys):
    with open(key, 'rb') as f:
        img_bytes = f.read()
    txn.put(key.encode(), img_bytes)
txn.commit()
env.close()

# test reading
test = False
if test:
    env = lmdb.open('small_dataset_lmdb')
    txn = env.begin(write=False)
    random.shuffle(keys)
    for key in keys:
        img_bytes = txn.get(key.encode())
        img_bytes = np.array(bytearray(img_bytes), dtype=np.uint8)
        if key.endswith('.jpg'):
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        elif key.endswith('.png'):
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
        else:
            print('Unknown image type: %s' % key)
            exit()
        img = cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))
        cv2.imshow('show', img)
        key = cv2.waitKey(0)
        if 27 == key:
            break
    env.close()
