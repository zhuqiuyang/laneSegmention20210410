
class Config(object):
    # model config
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8

    # train config
    EPOCHS = 10
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 1e-3

    #config
    OPTIMIZER = 1 #0: sgd, 1: adam, 2: nadam
    ADJUST_LR_ENABLE = 1 #0: disable, 1: enable
    LOSS_TYPE = 1 #0: soft max + cross entropy, 1: dice loss