import os
import random
from tqdm import tqdm

def write_csv(data_list, csv_fn):
    with open(csv_fn, 'w') as f:
        for image_path, label_path in tqdm(data_list, 'Write csv'):
            f.write('%s, %s\n' % (image_path, label_path))

image_list = []
label_list = []

image_dirs = ['../dataset/round1/train/Road%02d/ColorImage_road%02d' % (i, i) for i in range(2, 5)] + \
             ['../dataset/round2/train/ColorImage_road%02d' % i for i in range(2, 5)]
label_dirs = ['../dataset/round1/train/Gray_Label/Label_road%02d' % i for i in range(2, 5)] + \
             ['../dataset/round2/train/Gray_Label/Label_road%02d' % i for i in range(2, 5)]
image_label_dirs = list(zip(image_dirs, label_dirs))
out_dir = 'data_list'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for image_dir, label_dir in tqdm(image_label_dirs, desc='Read paths'):
    image_root = os.path.join(image_dir, 'ColorImage')
    label_root = os.path.join(label_dir, 'Label')
    # Record
    for record_folder in sorted(os.listdir(image_root)):
        record_path = os.path.join(image_root, record_folder)
        label_record_path = os.path.join(label_root, record_folder)
        assert os.path.exists(label_record_path)
        # Camera
        for camera_folder in sorted(os.listdir(record_path)):
            camera_path = os.path.join(record_path, camera_folder)
            label_camera_path = os.path.join(label_record_path, camera_folder)
            assert os.path.exists(label_camera_path)
            # Image
            for image_fn in sorted(os.listdir(camera_path)):
                image_path = os.path.join(camera_path, image_fn)
                label_path = os.path.join(label_camera_path, image_fn[:-4] + '_bin.png')
                assert os.path.exists(label_path)
                image_list.append(image_path)
                label_list.append(label_path)


assert len(image_list) == len(label_list), \
       "The length of image dataset is {}, and label is {}".format(len(image_list), len(label_list))
image_label_list = list(zip(image_list, label_list))

train_list = []
val_list = []
val_keywords = ['Record001', 'Record002', 'Record003']
for image_path, label_path in tqdm(image_label_list, desc='Split train/val'):
    is_val = False
    for val_filter in val_keywords:
        if val_filter in image_path:
            val_list.append((image_path, label_path))
            is_val = True
            break
    if not is_val:
        train_list.append((image_path, label_path))
write_csv(train_list, os.path.join(out_dir, 'train.csv'))
write_csv(val_list, os.path.join(out_dir, 'val.csv'))


#total_length = len(image_list)
#eighth_part = int(total_length*0.8)
#random.shuffle(image_label_list)
#
#train_list = image_label_list[:eighth_part]
#val_list = image_label_list[eighth_part:]
#
#write_csv(train_list, os.path.join(out_dir, 'train.csv'))
#write_csv(val_list, os.path.join(out_dir, 'val.csv'))
