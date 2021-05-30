# 数据集格式转换说明
## 原始数据分析
### 数据存放路径
```
small_dataset
    ├─ Gray_Label
    │     ├─ Label_road02
    │     │      └─ Label
    │     │          ├─ Record001
    │     │          │     ├─ Camera5
    │     │          │     │     ├─ 170927_063820123_Camera_5_bin.png
    │     │          │     │     └─ 170927_063843458_Camera_5_bin.png
    │     │          │     └─ Camera6
    │     │          │     │     └─ 170927_063844445_Camera_6_bin.png
    │     │          └─ ...
    │     ├─ Label_road03
    │     │      └─ ...
    │     └─ ...
    ├─ Road02
    │     ├─ ColorImage_road02
    │     │      └─ ColorImage
    │     │          ├─ Record001
    │     │          │     ├─ Camera5
    │     │          │     │     ├─ 170927_063820123_Camera_5.jpg
    │     │          │     │     └─ 170927_063843458_Camera_5.png
    │     │          │     └─ Camera6
    │     │          │     │     └─ 170927_063844445_Camera_6.png
    │     │          └─ ...
    │     └─Labels_road02
    │            └─ Label
    │                ├─ Record001
    │                │     ├─ Camera5
    │                │     │     ├─ 170927_063820123_Camera_5_bin.png
    │                │     │     └─ 170927_063843458_Camera_5_bin.png
    │                │     └─ Camera6
    │                │     │     └─ 170927_063844445_Camera_6_bin.png
    │                └─ ...
    ├─ Road03
    │     ├─ ...
    │     └─ ...
    └─ Road04
          ├─ ...
          └─ ...
```
### 数据目录
在`data_list`目录中存有`train.csv`和`val.csv`,分别存放训练数据目录与验证数据目录：
|  image_dir   |  label_dir  |
|  ----  | ----  |
| small_dataset/Road03/ColorImage_road03/ColorImage/Record002/Camera 6/171206_025922279_Camera_6.jpg  | small_dataset/Gray_Label/Label_road03/Label/Record002/Camera 6/171206_025922279_Camera_6_bin.png |
| small_dataset/Road03/ColorImage_road03/ColorImage/Record004/Camera 5/171206_030217196_Camera_5.jpg  | small_dataset/Gray_Label/Label_road03/Label/Record004/Camera 5/171206_030217196_Camera_5_bin.png |
**注：每行数据的image_dir与label_dir之间以`, `逗号分隔符+空格进行分隔，label_dir末尾有`\n`换行符进行换行**

## 原始数据转LMDB方法
### 思路
由于当前问题是一个语义分割问题，data与label均为图片（.png），因此可以依照`data_list`目录中的目录文件，依次以二进制格式读取图片，并为每一张图片（data与label）赋予一个名称，这样即可将这张图片以{key(名称)：value(二进制内容)}的字典格式存储到lmdb文件中，与此同时，在建立一个新csv文件用于存储图片名称，以供后模型训练时调取文件所用。
### 实现
```
def img_data_2_lmdb_data(csv_dir, lmdb_dir, name_csv_dir):
    '''
    根据图片汇总CSV文件，将图片存入LMDB结构中。
    '''
    # 创建环境（lmdb_dir）
    # 预制数据字典
    # 打开数据目录csv文件（csv_dir）
        # 遍历csv中的每一行
            # 获取两个路径,并对label路径进行简单整理（去除开头的空格和最后的换行符）
            # 为这两个文件命名，名字即为图片文件的名称（去除.png）
            # 以二进制格式打开这两张图片
            # 将图片名称作为key，二进制内容作为value，存储至字典中
            # 将图片名称写入csv文件保存（name_csv_dir）
            # 将数据字典推入lmdb文件中
    # 关闭lmdb环境
```
详见`make_lmdb.py`。

## dataloader模块从lmdb中获取数据的方法
### 思路
从lmdb中获取数据与逐个图片文件读取没有本质上的区别，都是从csv文件中获取总数据目录，而后根据目录逐一读取数据，只不过是从磁盘中一张一张的图片，还是从lmdb中用键找值的区别。
### 实现
详见`image_process.py`。
### 验证
详见`test_dataloader.py`。