import math
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

bn_mom = 0.0003

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_path_rate)
        )

    def forward(self, x):
        skip = x = self.conv1(x)
        x = self.conv2(x)
        return x+skip

class Block(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0.1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, drop_path_rate)
        self.hook_layer = None

    def forward(self, x):
        x = self.maxpool(x)
        self.hook_layer = x = self.conv(x)
        return x


class Resnet(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self,  drop_path_rate=0.1):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Resnet, self).__init__()

        self.inc = DoubleConv(3, 64, drop_path_rate)
        self.block1 = Block(64, 128, drop_path_rate)
        self.block2 = Block(128, 256, drop_path_rate)
        self.block3 = Block(256, 512, drop_path_rate)
        self.block4 = Block(512, 512, drop_path_rate)

        self.num_features = 512
        self.embed_dim = 64


    def forward(self, input):
        self.layers = []
        x1 = self.inc(input)
        x2 = self.block1(x1)
        low_featrue_layer = self.block1.hook_layer
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        return low_featrue_layer, x5


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from thop import profile
    from pytorch_model_summary import summary
    import time

    print(torch.__version__)
    net = Resnet().cuda()
    # net = swin_base_patch4_window12_384().cuda()
    import torchsummary

    # torchsummary.summary(net)
    print(net)
    image = torch.rand(1, 3, 512, 512).cuda()
    # time_step=torch.tensor([999] * 1, device="cuda")
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # f, p = profile(net, inputs=(image, time_step))

    f, p = profile(net, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.size)