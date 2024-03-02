import torch
import torch.nn as nn
import math
from models.utils import AvgPoolConv


cfg = {
    'VGG11': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_class=100, use_bn=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], use_bn)
        #如果需要vgg11或更小的模型需修改512
        self.classifier = nn.Linear(512, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, AvgPoolConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, use_bn=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [AvgPoolConv(kernel_size=2, stride=2, input_channel=in_channels)]
            else:
                # layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                #            nn.BatchNorm2d(x) if use_bn else nn.Dropout(0.25),
                #            nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x) ,
                           nn.ReLU(inplace=True),
                           # nn.Dropout(0.25)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == '__main__':
        net = VGG('VGG16')
        # x = torch.randn(2, 3, 32, 32)
        # y = net(x)
        print(net)
        # for name, immediate_child_module in net.named_children():
        #     for n ,m in immediate_child_module.named_children():
        #         for i,j in m.named_children():
        #             print(j)
        #         break
        # for name ,module in net.named_modules():
        #     if isinstance(module,(nn.ReLU, nn.ReLU6)):
        #         print(name)
        for name,p in net.named_parameters():
            print(name,p)