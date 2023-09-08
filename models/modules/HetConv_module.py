from torch import nn

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, padding=1, dilation=1, p=4):
        super(HetConv, self).__init__()
        if in_channels % p != 0:
            raise ValueError('in_channels must be divisible by p')
        if out_channels % p != 0:
            raise ValueError('out_channels must be divisible by p')
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, groups=p, bias=bias)
        self.conv1x1_ = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, bias=bias)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) - self.conv1x1_(x)