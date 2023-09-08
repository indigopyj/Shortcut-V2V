import math

import torch
import torch.nn as nn
from models.modules.HetConv_module import HetConv


class LocalOffsetGenerator(nn.Module):
    def __init__(self, opt, input_nc, ngf, output_nc, n_layers=1):
        super(LocalOffsetGenerator, self).__init__()
        self.opt=opt
        self.model = []
        for l in range(n_layers):
            if l != 0: input_nc = ngf
            self.model += [HetConv(input_nc, ngf),
                        nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.offset_mask_conv = nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
        
    def forward(self, img1, img2):
        output = self.model(torch.cat((img2, img1), dim=1))
        offset_mask = self.offset_mask_conv(output)
        return offset_mask
