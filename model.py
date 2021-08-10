import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Bottleneck(nn.Module):
    # 每个Stage中输入与输出通道的倍数
    expansion = 4

    '''
    in_channel:每个block的输入通道数
    mid_channel:每个block中中间层的通道数，即为输出通道数点的1/4
    mid_plane*self.extention：输出的维度
    stride:每个block中第一个Conv2d的滑步数
    downsample:下采样默认空
    '''

    def __init__(self, in_channel, mid_channel, stride = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, mid_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channel * self.expansion)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 是否直连（如果是Identity block就是直连；如果是Conv Block就需要对参差边进行卷积，改变通道数和size）
        if self.downsample is not None:
            identity = self.downsample(x)
        # 卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64  # 输入特征图的深度(经过初始的maxpooling之后的特征图)

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,  # 注意哦，self.in_channel是作为第一个卷积层的输出个数
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化操作
        self.layer1 = self.make_layer(block, 64, blocks_num[0])  # 第一个残差层 Conv_2
        self.layer2 = self.make_layer(block, 128, blocks_num[1], stride=2)  # 第二个残差层 Conv_3
        self.layer3 = self.make_layer(block, 256, blocks_num[2], stride=2)  # 第三个残差层 Conv_4
        self.layer4 = self.make_layer(block, 512, blocks_num[3], stride=2)  # 第四个残差层 Conv_5

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        # conv+bn+relu+maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def make_layer(self, block, channel, block_num, stride=1):
        block_list = []

        # 判断要不要加downsample模块
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channels=channel * block.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        # Conv Block
        Conv_block = block(self.in_channel, channel, stride=stride, downsample=downsample)
        block_list.append(Conv_block)
        self.in_channel = channel * block.expansion

        # Identity Block
        for _ in range(1, block_num):
            block_list.append(block(self.in_channel, channel))
        return nn.Sequential(*block_list)


ResNet50 = ResNet(Bottleneck, [3, 4, 6, 3], 1000)
# print(ResNet50)
# writer = SummaryWriter("logss")
# writer.add_graph(ResNet50, torch.ones(1, 3, 224, 224))
# writer.close()
