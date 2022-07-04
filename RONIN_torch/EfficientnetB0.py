"""
This is the 1-D  version of EfficientNetB0
Original paper is "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
Link: https://arxiv.org/abs/1905.11946

The implementation in https://github.com/AnjieCheng/MnasNet-PyTorch/blob/master/MnasNet.py has been modified.
A simple code has been added to calculate the number of FLOPs and parameters
from https://github.com/1adrianb/pytorch-estimate-flops.
"""

from torch.autograd import Variable
import torch.nn as nn
import torch
import math


# siply use torch.nn.SiLU for recent pytorch versions. My pytorch was 1.4, I had to define it myself
def my_swish(input):
    return input * torch.sigmoid(input)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return my_swish(input)


swish = Swish()


def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm1d(oup),
        # nn.ReLU6(inplace=True),
        swish
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm1d(oup),
        # nn.ReLU6(inplace=True)
        swish
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv1d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm1d(inp * expand_ratio),
            # nn.ReLU6(inplace=True),
            swish,
            # dw
            nn.Conv1d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm1d(inp * expand_ratio),
            # nn.ReLU6(inplace=True),
            swish,
            # pw-linear
            nn.Conv1d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNetB0(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(EfficientNetB0, self).__init__()

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [1, 16, 1, 1, 3],  # -> 56x56
            [6, 24, 2, 2, 3],  # -> 28x28
            [6, 40, 2, 2, 5],  # -> 14x14
            [6, 80, 3, 2, 3],  # -> 14x14
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layer
        self.features = [Conv_3x3(6, input_channel, 2)]
        # input_channel = 16

        # building inverted residual blocks (MBConv)
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, k))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                input_channel = output_channel

        # building last several layers
        self.features.append(Conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool1d(1))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = EfficientNetB0(n_class=2)
    print(net)
    x_image = Variable(torch.randn(1, 6, 200))
    y = net(x_image)
    print(y)
    inp = torch.rand(1, 6, 200)
    from pthflops import count_ops

    # Count the number of FLOPs
    count_ops(net, inp)
    print(net.get_num_params())