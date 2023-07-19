import inspect
import os

import fire
import torch
import torchvision
import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchnet.meter import AverageValueMeter

from config import ExtDefaultConfig, DefaultConfig
from data import AnimeFaces128Dataset
from models import GeneratorNet, DiscriminatorNet

opt = ExtDefaultConfig()


def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """

    opt.parse(**kwargs)
    opt.print_attr()

    if opt.vis:
        from utils import Visualizer
        vis = Visualizer(opt.visdom_env)

    # 数据
    dataset = AnimeFaces128Dataset(opt.image_size, opt.data_path)
    # dataset = ImageFolder(opt.data_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                            drop_last=True)

    # Net
    net_g, net_d = _get_model()

    # 定义优化器
    optimizer_g = torch.optim.Adam(net_g.parameters(), opt.lr_g, betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(net_d.parameters(), opt.lr_d, betas=(opt.beta1, 0.999))

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    # true_labels = torch.ones(opt.batch_size).to(opt.device)
    # fake_labels = torch.zeros(opt.batch_size).to(opt.device)
    fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(opt.device)
    noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(opt.device)

    error_meter_g = AverageValueMeter()
    error_meter_d = AverageValueMeter()

    for epoch in range(1, opt.max_epoch + 1):
        epoch_flag = f"{epoch}/{opt.max_epoch}"

        net_g.train(), net_d.train()
        _train_epoch(epoch_flag, net_d, net_g, optimizer_d, optimizer_g, dataloader, error_meter_d, error_meter_g,
                     fix_noises, noises, vis)

        if epoch % opt.save_every == 0:
            # 保存图片、模型
            fix_fake_imgs = net_g(fix_noises).detach()
            if not os.path.exists(opt.train_save_path) or not os.path.isdir(opt.train_save_path):
                os.mkdir(opt.train_save_path)
            file_path = os.path.join(opt.train_save_path, str(epoch) + ".png")
            torchvision.utils.save_image(fix_fake_imgs.data[:64], file_path, normalize=True, value_range=(-1, 1))

            net_g.save('GeneratorNet_%s.pth' % epoch)
            net_d.save('DiscriminatorNet_%s.pth' % epoch)

            error_meter_g.reset()
            error_meter_d.reset()


def _train_epoch(epoch_flag, net_d, net_g, optimizer_d, optimizer_g, dataloader, error_meter_d, error_meter_g,
                 fix_noises, noises, vis):
    progress = tqdm.tqdm(dataloader, desc=f"Train... [Epoch {epoch_flag}]")
    for i, (img, _) in enumerate(progress, 1):
        real_img = img.to(opt.device)

        # 训练判别器
        if i % opt.d_every == 0:
            optimizer_d.zero_grad()

            # 尽可能的把真图片判别为正确，尽可能把假图片判别为错误
            r_preds = net_d(real_img)
            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            with torch.no_grad():
                fake_img = net_g(noises)  # 现在不训练生成器
            f_preds = net_d(fake_img)

            # 计算损失
            r_f_diff = (r_preds - f_preds.mean()).clamp(max=1)
            f_r_diff = (f_preds - r_preds.mean()).clamp(min=-1)
            loss_d = (1 - r_f_diff).mean() + (1 + f_r_diff).mean()

            loss_d.backward()
            optimizer_d.step()
            error_meter_d.add(loss_d.item())

        # 训练生成器
        if i % opt.g_every == 0:
            optimizer_g.zero_grad()

            # 尽可能提高生成的图像被判别为真的概率
            noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
            fake_img = net_g(noises)
            f_preds = net_d(fake_img)
            r_preds = net_d(real_img)

            # 计算损失
            r_f_diff = F.relu(r_preds - f_preds.mean())
            f_r_diff = F.relu(f_preds - r_preds.mean())
            error_g = (1 + r_f_diff).mean() + (1 - f_r_diff).mean()

            error_g.backward()
            optimizer_g.step()
            error_meter_g.add(error_g.item())

        if opt.vis and i % opt.plot_every == 0:
            # 可视化
            fix_fake_imgs = net_g(fix_noises).detach()
            vis.images(fix_fake_imgs.cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
            vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
            vis.plot('error_g', error_meter_g.value()[0])
            vis.plot('error_d', error_meter_d.value()[0])


@torch.no_grad()
def generate(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    opt.parse(**kwargs)
    opt.print_attr()

    # Net
    net_g, net_d = _get_model()
    net_g.eval(), net_d.eval()

    noises = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(opt.device)

    # 生成图片，并计算图片在判别器的分数
    with torch.no_grad():
        fake_img = net_g(noises)
        scores = net_d(fake_img)

        # 挑选最好的某几张
        indexs = scores.topk(opt.gen_num)[1]
        result = []
        for i in indexs:
            result.append(fake_img.data[i])
        # 保存图片
        torchvision.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, value_range=(-1, 1))


def _get_model():
    net_g, net_d = GeneratorNet(opt.ngf, opt.nz), DiscriminatorNet(opt.ndf)
    if opt.netg_path:
        net_g.load(opt.netg_path)
    if opt.netd_path:
        net_d.load(opt.netd_path)
    return net_g.to(opt.device), net_d.to(opt.device)


def help():
    """
    打印帮助信息
    :return:
    """
    print("""
    usage: python {0} <function> [--args=value,]
    <function> := train | generate | help
    examples:
            python {0} train --visdom_env='image-generation' --data-path='dataset/AnimeFaces128' --max-epoch=200 
            python {0} generate --netg-path='checkpoints/GeneratorNet_200.pth' --netd-path='checkpoints/DiscriminatorNet_200.pth' --gen-num=64
            python {0} help
    avaiable args:
    """.format("main.py"))

    source = inspect.getsource(DefaultConfig)
    print(source)


if __name__ == '__main__':
    fire.Fire()
