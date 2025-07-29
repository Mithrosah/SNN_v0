import torch
import torch.nn as nn

import layers


class SMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # self.conv1 = layers.SConv2d(1, 27, 3, 1, 1)
        # self.actv1 = layers.SActv(5)
        # self.pool1 = layers.SAvgPool2d(3, 2, 1)
        #
        # self.conv2 = layers.SConv2d(27, 81, 3, 1, 1)
        # self.actv2 = layers.SActv(6)
        # self.pool2 = layers.SAvgPool2d(3, 2, 1)
        #
        # self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(81*7*7, num_classes)

        self.conv1 = nn.Conv2d(1, 27, 3, 1, 1)
        self.actv1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(3, 2, 1)

        self.conv2 = layers.SConv2d(27, 81, 3, 1, 1)
        self.actv2 = layers.SActv(6)
        self.pool2 = layers.SAvgPool2d(3, 2, 1)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(81*7*7, num_classes)



    def forward(self, x):
        x = self.pool1(self.actv1(self.conv1(x)))
        x = self.pool2(self.actv2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 27, 3, 1, 1)
        self.actv1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(3, 2, 1)

        self.conv2 = nn.Conv2d(27, 81, 3, 1, 1)
        self.actv2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(3, 2, 1)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(81*7*7, num_classes)

    def forward(self, x):
        x = self.pool1(self.actv1(self.conv1(x)))
        x = self.pool2(self.actv2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = SMNIST()
    x = torch.randn(4, 1, 28, 28)
    print(model(x).shape)