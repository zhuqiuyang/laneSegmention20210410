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
CV名企课-车道线分割-WEEK1 :  转置卷积与FCN
https://gitee.com/mingminglaoshi/lane-segmention20210410
Pipeline:
0.双线性上采样作业回顾
1.转置卷积详解与反向传播
2.全卷积网略(FCN)

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
