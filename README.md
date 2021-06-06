# laneSegmention20210410
  - 开讲日期20210411-上午9点
  - 以下文件都可以在https://gitee.com/mingminglaoshi/lane-segmention20210410     找到，若找不到联系，可以明明老师协助。微信13271929138
  - week1 
```
CV名企课-车道线分割-WEEK1 :  项目概述及上采样技术
https://gitee.com/mingminglaoshi/lane-segmention20210410
Pipeline:
0.师生相互了解
1.项目概述
2.卷积神经网络
3.上采样技术

作业：
   1. 安装python环境，安装numpy,pytorch等库、
   2. 跑通老师提供的双线性插值循环版本upsampling.ipynb 
   3. 用numpy实现双线性插值的矩阵版本（函数提示:np_marid,np.clip, np.expand_dims
   4. 测试双线性插值循环版和矩阵版本的运行时间和结果差异（参考陈老师提供的nearest的代码)
提交要求：
   1. 提交双线性插值矩阵版本的python代码（建议.py 或 .ipynb格式)
   2. 提交两个版本的运行时间和结果差异截图
week1文件夹说明：
   1. cat.png:upsampling.ipynb 使用的样例图片
   2. upsampling.ipynb 上采样参考代码
```
  - week2 
```
CV名企课-车道线分割-WEEK2 :  转置卷积与FCN
https://gitee.com/mingminglaoshi/lane-segmention20210410
Pipeline:
0.双线性上采样作业回顾
1.转置卷积详解与反向传播
2.全卷积网络(FCN)

作业：
   1. 参照代码week2/fcn.py,用pytorch实现FCN-4s(padding=100)
提交内容：
   1. FCN-4s的代码（建议.py格式和ipynb格式)
   2. FCN-4s中的crop的offset计算过程(建议word格式)

week2文件夹内容说明：
   1. fcn.py：FCN的参考资料
   2. upsampling_homework.py:双线性上采样实现方法。
   3. Long_Fuuly_Convolutional_Networks_2015_CVPR_paper.pdf:FCN论文
```
  - week3 
```
CV名企课-车道线分割-WEEK3 :  U-Net模型详解
https://gitee.com/mingminglaoshi/lane-segmention20210410
Pipeline:
0.FCN-4s作业回顾
1.U-Net详解
2.U-Net网络结构修改
3.扩展学习:U-Net++

作业：
   1. 对老师提供的 U-Net++L2代码进行修改，backbone替换为ResNet,其他卷积块替换为ResNet中的BasicBlock。按照作业中的注释要求进行修改。
提交内容：
   1. 提交修改后的代码（建议.py格式和ipynb格式）
   2. 将backbone分别替换为resnet34和resnet50,将打印出的结果截图提交

week3文件夹内容说明：
   1. week2参考答案/FCN_4s.py：FCN实现4s的代码参考
   2. 作业：week3的作业代码
   3. 论文：week3涉及到的UNet,ResNet,hourglass以及pix2pix的论文。
   4. 课上代码：课堂上涉及到的UNet.py,ResNet.py相关代码。
补充内容：
   1. U-Net论文解析：https://zhuanlan.zhihu.com/p/370931792
   2. ResNet34模型计算过程详解：https://zhuanlan.zhihu.com/p/370931435
   3. 最佳backbone:resnet的几种变形：https://zhuanlan.zhihu.com/p/370930808
   4. U- Net论文带读以及代码带写：https://www.bilibili.com/video/BV1KK4y1A7TD
```

  - week4 
```
CV名企课-车道线分割-WEEK4 :  DeepLab模型详解
https://gitee.com/mingminglaoshi/lane-segmention20210410

Pipeline:
0.UNet++作业回顾
1.膨胀卷积
2.DeepLabV1
3.DeepLabV2
4.DeepLabV3
5.DeepLabV3+
6.DeepLabV3+代码实战

作业：
   1. 基于老师提供的aspp.py文件，补全aspp模块。
   2. 计算backbone为resnet50_atrous时，ASPP五个分支各自感受野的大小（要求计算output_stride=8,和output_stride=16的情况)
提交内容：
   1. 提交修改后的aspp的完整代码（建议.py格式和ipynb格式）
   2. 感受野的大小计算过程以及结果(建议txt,word或者pad格式)

week4文件夹内容说明：
   1. week4/week3参考答案/resnet_unetpp_homework.py：week3的UNetpp的L2级别的代码参考
   2. week4/aspp.py:week4的作业代码
   3. 论文：week4涉及到的deeplabv1,v2,v3,v3+等论文。
   4. 课上代码：课堂上涉及到的deeplabv3plus.py
补充内容：
   1. 关于空洞卷积的讨论：https://zhuanlan.zhihu.com/p/372753977
   2. Deeplabv1论文解析：https://zhuanlan.zhihu.com/p/373825061
   3. ResNet50模型计算过程详解：https://zhuanlan.zhihu.com/p/374448655
   4. 条件随机场CRF与图像分割：https://zhuanlan.zhihu.com/p/372759285
   5. DeepLabv1与VGG-16模型的感受野计算：https://zhuanlan.zhihu.com/p/373639725
```

- week5 
```
CV名企课-车道线分割-WEEK5 :  数据处理
https://gitee.com/mingminglaoshi/lane-segmention20210410

Pipeline:
0.ASPP,以及DeeplabV3+感受野作业回顾
1.数据集制作
2.数据处理
3.数据加载
4.数据闭环与主动学习

作业：
   1. 将small_dataset制作成LMDB格式的数据集
   2. 基于老师提供的代码，将LaneDataset 改写为从LMDB中读取数据
   [选做]3. 使用小批量数据集进行 deeplabv3+模型的从零开始训练，使用交叉熵，SGD进行训练。

提交内容：
   1. README.txt - LMDB制作方法概述(Key的选择，数据解析方式等)
   2. make_lmdb.py - LMDB的制作脚本
   3. image_process.py - 其中LaneDataset读取方式改为LMDB

week5文件夹内容说明：
   1. week5/class_code：week5课上代码，主要涉及颜色变换，投影变换等图片增广的示范程序。
   2. week5/homework:作业参考代码。小批量数据集的数据预处理与数据增广，数据加载程序。作业可在此代码的基础上更改。
   3. week4_homework_anser:ASPP代码以及deeplabv3+感受野计算参考答案。

补充内容：
   1. 感受野前向计算，后向计算的公式以及公式推导：
      前向计算：https://zhuanlan.zhihu.com/p/375100687
      后向计算：待更新在https://gitee.com/mingminglaoshi/lane-segmention20210410/blob/master/README.md
   2. 车道线检测经典常用数据集整理：待更新在https://gitee.com/mingminglaoshi/lane-segmention20210410/blob/master/README.md
   3. LMDB的生成与读取:https://www.cnblogs.com/houjun/p/10484945.html
   4. 图像数据增强包IAA：https://blog.csdn.net/qq_36552489/article/details/107645788
   5. IMDB2IMG方法,IMG2IMDB方法：https://www.jianshu.com/p/ef84715e0fdc

```
- week6
```
CV名企课-车道线分割-WEEK6 :  模型训练
https://gitee.com/mingminglaoshi/lane-segmention20210410
Pipeline:
0. 将数据做成LMDB的作业回顾
1. 训练流程概述
2. 参数初始化
3. 优化器
4. 学习率调整
5. 损失函数
6. 代码详解

作业：
   1. 下载数据集，配置GPU训练环境，跑通老师提供的代码作为Baseline
   2. 在Baseline基础上，尝试1-3种优化方案，提升模型准确率
        提示：可以从优化器，学习率策略，损失函数，样本均衡，数据扩增，多分辨率，模型融合等方面优化
提交内容：
   1. Baseline跑通截图
   2. README.txt -- 说明优化方案的设计思路及实施细节，同时指明相关代码的文件名及行数（如：在train.py的23-26行进行了第一种优化)
   3. 修改后的项目代码（只提交代码，不要提交模型文件)
   4. 验证集最优结果截图
文件夹内容说明：
   1. week6/class_code_homework:课上代码文件夹，也就是我们需要跑通的作业代码所在文件夹
   2. week6/class_code_homework/train.py：训练代码，含训练逻辑部分
   3. week6/class_code_homework/predict.py: 测试代码，含加载模型，利用模型前向计算，给前向计算结果染色，并保存图片。
   4. week6/utils:此文件夹内含有数据处理程序，loss函数定义等。
   5. week6/model:此文件夹为FCN,UNET,deeplabv3p等模型声明，以及初始化部分。
   6. week5_answer_vip:每个图片保存一个data,LMDB数据处理程序参考
   7. week5_answer_kkb:所有图片保存一个data,LMDB数据处理程序参考
补充内容:
   1. 数据集下载路径：链接: https://pan.baidu.com/s/1x0B37jvTkrDikGITK_efcQ 密码: of61
   2. 平均分布均值和方差的推导：https://zhuanlan.zhihu.com/p/376497110
   3. 梯度下降算法几种变种的补充材料：https://www.cnblogs.com/neopenx/p/4768388.html
   4. LMDB字节数的计算方法：map_size 指定创建的新数据库所需磁盘空间的最小值，1099511627776B＝１T。可以在这里https://cunchu.51240.com存储单位换算，也可以看下图：
```
![输入图片说明](https://images.gitee.com/uploads/images/2021/0530/173303_79a4f2a2_8791625.png "屏幕截图.png")


- week7 
```
CV名企课-车道线分割-WEEK7: 模型的评价与部署
https://gitee.com/mingminglaoshi/lane-segmention20210410
Pipeline:
1. 模型训练作业回顾
2. 语义分割评价指标
3. 模型测试
4. 服务端部署简介
5. 移动端部署简介

作业:
   1. 跑通Flask的demo,并将模型替换为自己在week6训练的模型
   2. 学习MNN,编译并跑通老师提供的PortraitNet代码(Windows/Linux均可)，http://github.com/alibaba/MNN,https://www.yuque.com/mnn/cn
   3. [选做] 将自己训练的模型进行转换(pytorch->onnx->MNN),并修改PortraitNet代码为车道线分割代码，编译运行(Windows/Linux均可)

提交内容:
   1. Flask_demo.py跑通截图
   2. PortraitNet代码跑通截图
   3. ReadMe.txt---描述PortraitNet编译跑通的具体步骤，以及遇到的难点和解决办法

文件夹内容说明：
   1. week7/week6_answer： week6作业的优秀学员作业，作为参考
   2. week7/lane_segmentation: 内含flask_demo.py,这个是服务端部署的一个简单的demo,需要运行起来然后截图
   3. week7/lane_segmentation/flask_demo.py,作业1要求运行成功的代码
   4. week7/lane_segmentation/templates/index.html, 服务器上的主页
   5. week7/lane_segmentation/templates/show_result.html, 服务器上的结果显示页面
   6. week7/MNN: MNN的github源码，readme等，windows编译包，以及测试代码main.cpp 
   7. week7/MNN/MNN/PortraitNet.mnn 模型参数以及模型结构保存文件。
   8. week8/NMM/MNN/test/compile.bat,  windows上使用的编译命令。

补充内容:
   1. flask_demo.py 环境准备工作以及运行视频：https://zhuanlan.zhihu.com/p/378379199
   2. MNN在mac/linux上的编译，运行步骤以及相关视频[待更新]
   3. CircleLR为何可用比较小的学习率【待更新】
   4. 模型量化相关资料整理【待更新】
```
