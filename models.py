import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, ch=32, dim_c=1, n_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(dim_c, ch, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(ch, ch * 2, kernel_size=3, stride=1)
        self.d1 = nn.Dropout2d(0.25)
        self.d2 = nn.Dropout2d(0.5)
        self.l1 = nn.Linear(9216, 128)
        self.l2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.d1(x)
        x = torch.flatten(x, 1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = self.l2(x)
        out = F.log_softmax(x, dim=1)
        return out
