from torch import nn

from .basic_module import BasicModule


class GeneratorNet(BasicModule):

    def __init__(self, ngf, nz):
        """
        生成器
        :param ngf: 生成器feature map数
        :param nz: 噪声维度
        """
        super(GeneratorNet, self).__init__()

        def tcnn_block(ins, outs, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(ins, outs, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(outs),
                nn.ReLU(True)
            )

        self.tcnn_block1 = tcnn_block(nz, ngf * 8, 4, 1, 0)  # (ngf*8) x 4 x 4
        self.tcnn_block2 = tcnn_block(ngf * 8, ngf * 4, 4, 2, 1)  # (ngf*4) x 8 x 8
        self.tcnn_block3 = tcnn_block(ngf * 4, ngf * 2, 4, 2, 1)  # (ngf*2) x 16 x 16
        self.tcnn_block4 = tcnn_block(ngf * 2, ngf, 4, 2, 1)  # (ngf) x 32 x 32
        self.tcnn_block5 = nn.Sequential(nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False), nn.Tanh())  # 3 x 96 x 96

    def forward(self, X):
        out = self.tcnn_block1(X)
        out = self.tcnn_block2(out)
        out = self.tcnn_block3(out)
        out = self.tcnn_block4(out)
        out = self.tcnn_block5(out)
        return out


class DiscriminatorNet(BasicModule):

    def __init__(self, ndf):
        """
        判别器
        :param ndf: 判别器feature map数
        """
        super(DiscriminatorNet, self).__init__()

        def cnn_block(ins, outs, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(ins, outs, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(outs),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # 输入 3 x 96 x 96
        self.cnn_block1 = nn.Sequential(nn.Conv2d(3, ndf, 5, 3, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))  # 输出 (ndf) x 32 x 32
        self.cnn_block2 = cnn_block(ndf, ndf * 2, 4, 2, 1)  # 输出 (ndf*2) x 16 x 16
        self.cnn_block3 = cnn_block(ndf * 2, ndf * 4, 4, 2, 1)  # 输出 (ndf*4) x 8 x 8
        self.cnn_block4 = cnn_block(ndf * 4, ndf * 8, 4, 2, 1)  # 输出 (ndf*8) x 4 x 4
        self.cnn_block5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)  # 输出 (ndf*8) x 1

    def forward(self, X):
        out = self.cnn_block1(X)
        out = self.cnn_block2(out)
        out = self.cnn_block3(out)
        out = self.cnn_block4(out)
        out = self.cnn_block5(out)
        return out.view(-1)

