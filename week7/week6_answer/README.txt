我在config.py中增加了三个选项：
    OPTIMIZER = 1 #0: sgd, 1: adam, 2: nadam
    ADJUST_LR_ENABLE = 1 #0: disable, 1: enable
    LOSS_TYPE = 1 #0: soft max + cross entropy, 1: dice loss
	
我一共实验了3种方法：SGD, NADAM, Dice loss。

(1) baseline的参数为：OPTIMIZER=1, ADJUST_LR_ENABLE=1, LOSS_TYPE=0
运行结果为：miou=0.4833。运行时的log为：log_base.txt。

(2) 用SGD作为优化器，不调整学习率。参数为：OPTIMIZER=0, ADJUST_LR_ENABLE=0, LOSS_TYPE=0
运行结果为：miou=0.4804。运行时的log为：log_sgd.txt。

(3) 用NADAM作为优化器，不调整学习率。参数为：OPTIMIZER=2, ADJUST_LR_ENABLE=0, LOSS_TYPE=0
NADAM的代码是从github上搜到的。运行结果很差，不太正常。后面需要查找原因。运行时的log在log_nadam.txt。

(4) Dice loss。参数为：OPTIMIZER=1, ADJUST_LR_ENABLE=1, LOSS_TYPE=1
运行结果很差，不太正常。后面需要查找原因。运行时的log在log_dice.txt。

代码在codes文件夹下：
以上三个配置参数的定义在config.py的第16~19行。
sgd的调用在：train.py的第117~118行。
adam的定义在utils/n_adam.py中，调用在train.py的第121~122行。
ADJUST_LR_ENABLE的使用在：train.py的第124~125行。
Dice loss的实现在utils/loss.py中的第24~43行，调用在train.py中的第29行。