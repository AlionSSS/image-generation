import warnings

import torch.cuda


class DefaultConfig(object):
    vis = True  # 是否使用visdom可视化
    visdom_env = "image-generation"  # Visdom的环境
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    data_path = 'dataset/AnimeFaces128'  # 数据集存放路径
    num_workers = 4  # 加载数据的进程数
    image_size = 96  # 图片尺寸，默认96 * 96
    batch_size = 256

    max_epoch = 200
    lr_g = 2e-3  # 生成器的学习率
    lr_d = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 使用的设备，cpu或cuda

    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    train_save_path = 'train_img_tmp/'  # 训练时生成图片保存路径

    d_every = 1  # 每1个batch训练一次判别器
    g_every = 2  # 每2个batch训练一次生成器
    save_every = 10  # 每10个epoch保存一次模型
    netd_path = None  # 'checkpoints/DiscriminatorNet_200.pth'  # Discriminator 预训练模型
    netg_path = None  # 'checkpoints/GeneratorNet_200.pth'  # Generator 预训练模型

    # 测试时所用参数
    gen_img = 'result.png'
    # 从5000张生成的图片中保存最好的64张
    gen_search_num = 5000
    gen_num = 64
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


class ExtDefaultConfig(DefaultConfig):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f"Warning: opt has not attribute {k}")
            setattr(self, k, v)

    def print_attr(self):
        print("user config:")
        for k, v in self.__class__.__base__.__dict__.items():
            if not k.startswith("__"):
                print("\t" + k + " = " + str(getattr(self, k)))
