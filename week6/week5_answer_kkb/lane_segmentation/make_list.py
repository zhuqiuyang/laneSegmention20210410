import os
import random

image_list = []
label_list = []

image_dirs = ['small_dataset/Road%02d' % i for i in range(2, 5)]
label_dir = 'small_dataset/Gray_Label/'

for image_dir in image_dirs:
    road_idx = int(image_dir[-1])
    image_root = os.path.join(image_dir, 'ColorImage_road%02d/ColorImage' % road_idx)
    label_root = os.path.join(label_dir, 'Label_road%02d/Label' % road_idx)
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
total_length = len(image_list)
eighth_part = int(total_length*0.8)

image_label_list = list(zip(image_list, label_list))
random.shuffle(image_label_list)

train_list = image_label_list[:eighth_part]
val_list = image_label_list[eighth_part:]

def write_csv(data_list, csv_fn):
    with open(csv_fn, 'w') as f:
        for image_path, label_path in data_list:
            f.write('%s, %s\n' % (image_path, label_path))

out_dir = 'data_list'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
write_csv(train_list, os.path.join(out_dir, 'train.csv'))
write_csv(val_list, os.path.join(out_dir, 'val.csv'))
