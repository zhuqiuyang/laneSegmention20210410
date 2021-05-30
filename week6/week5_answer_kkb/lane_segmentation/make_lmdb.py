from tqdm import tqdm
import lmdb

def img_data_2_lmdb_data(csv_dir, lmdb_dir, name_csv_dir):
    '''
    根据图片汇总CSV文件，将图片存入LMDB结构中。
    '''
     # 创建环境、预制数据字典
    env = lmdb.open(lmdb_dir, map_size=1099511627776)
    data_dict = dict()
    # 为tqdm获取total值
    num_file = sum([1 for i in open(csv_dir, 'r')])

    with open(csv_dir) as file:
        # 遍历csv中的每一行
        for line in tqdm(file, total=num_file):
            # 获取两个路径,并对label路径进行简单整理（去除开头的空格和最后的换行符）
            image_path, label_path = line.split(',')
            label_path = label_path[1:-1]
            # 为这两个文件命名，名字即为图片文件的名称（去除.png）
            image_name = image_path.split('/')[-1].split('.')[0]
            label_name = label_path.split('/')[-1].split('.')[0]
            # 以二进制格式打开这两张图片
            with open(image_path, 'rb') as f:
                image = f.read()
            with open(label_path, 'rb') as f:
                label = f.read()
            # 将图片名称作为key，二进制内容作为value，存储至字典中
            data_dict['image_' + image_name] = image
            data_dict['label_' + label_name] = label
            # 将图片名称（key）写入csv文件保存
            with open(name_csv_dir, 'a') as f:
                f.write('image_' + image_name + ',' + 'label_' + label_name + '\n')
            # 将数据字典推入lmdb文件中
            with env.begin(write=True) as txn:
                for key, value in data_dict.items():
                    txn.put(key.encode(), value)
    # 关闭lmdb环境
    env.close()


if __name__ == '__main__':
    csv_dir = 'data_list/val.csv'
    lmdb_dir = 'LMDB'
    name_csv_dir = 'LMDB/val_list.csv'

    img_data_2_lmdb_data(csv_dir, lmdb_dir, name_csv_dir)

