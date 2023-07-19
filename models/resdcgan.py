from torch import nn
from torch.nn import functional as F

from .basic_module import BasicModule


class GResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.is_shortcut = stride > 1
        self.shortcut = None if not self.is_shortcut else self._shortcut(in_channels, out_channels, stride)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        # 当X的维度和out不一致时，需要用shortcut处理X
        out += X if not self.shortcut else self.shortcut(X)
        out = F.relu(out)
        return out

    def _shortcut(self, in_channels, out_channels, stride):
        # 放大一倍
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )


class GeneratorNet(BasicModule):

    def __init__(self, ngf, nz):
        """
        生成器
        :param ngf: 生成器feature map数
        :param nz: 噪声维度
        """
        super(GeneratorNet, self).__init__()
        self.pre = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )  # (ngf*8) x 4 x 4
        self.layer1 = self._make_layer(ngf * 8, ngf * 8, 3, 3, 1)  # (ngf*8) x 4 x 4
        self.layer2 = self._make_layer(ngf * 8, ngf * 4, 6, 4, 2)  # (ngf*4) x 8 x 8
        self.layer3 = self._make_layer(ngf * 4, ngf * 2, 4, 4, 2)  # (ngf*2) x 16 x 16
        self.layer4 = self._make_layer(ngf * 2, ngf, 3, 4, 2)  # (ngf) x 32 x 32
        self.tcnn_block5 = nn.Sequential(nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False), nn.Tanh())  # 3 x 96 x 96

    def _make_layer(self, in_channels, out_channels, block_num, kernel_size, stride):
        layers = [GResidualBlock(in_channels, out_channels, kernel_size, stride)]
        for _ in range(block_num - 1):
            layers.append(GResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.pre(X)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.tcnn_block5(out)
        return out


class DResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.is_shortcut = stride > 1
        self.shortcut = None if not self.is_shortcut else self._shortcut(in_channels, out_channels, stride)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        # 当X的维度和out不一致时，需要用shortcut处理X
        out += X if not self.shortcut else self.shortcut(X)
        out = F.relu(out)
        return out

    def _shortcut(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )


class DiscriminatorNet(BasicModule):

    def __init__(self, ndf):
        """
        判别器
        :param ndf: 判别器feature map数
        """
        super(DiscriminatorNet, self).__init__()

        # 输入 3 x 96 x 96
        self.pre = nn.Sequential(nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
                                 nn.LeakyReLU(0.2, inplace=True))  # 输出 (ndf) x 32 x 32
        self.layer1 = self._make_layer(ndf, ndf, 3, 1)  # 输出 (ndf) x 32 x 32
        self.layer2 = self._make_layer(ndf, ndf * 2, 4, 2)  # 输出 (ndf) x 16 x 16
        self.layer3 = self._make_layer(ndf * 2, ndf * 4, 6, 2)  # 输出 (ndf) x 8 x 8
        self.layer4 = self._make_layer(ndf * 4, ndf * 8, 3, 2)  # 输出 (ndf) x 4 x 4
        self.cnn_block5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def _make_layer(self, in_channels, out_channels, block_num, stride):
        layers = [DResidualBlock(in_channels, out_channels, stride)]
        for _ in range(block_num - 1):
            layers.append(DResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.pre(X)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.cnn_block5(out)
        return out.view(-1)
