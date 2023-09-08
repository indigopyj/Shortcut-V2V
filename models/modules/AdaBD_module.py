import torch
import torchvision.ops
from torch import nn
import math
class AdaBD(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 num_deformable_groups=1,
                 stride=1,
                 padding=1,
                 bias=False,
                 dilation=1,
                 opt=None):

        super(AdaBD, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.num_deformable_groups = num_deformable_groups
        self.opt=opt
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=self.kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

            
    

    def forward(self, ref_act, new_act, offset_mask, isTrain=False):
    #def forward(self, ref_act, new_act, offset_mask):
        unmasked_ref_act = None
        offset = offset_mask[:, :2 * self.kernel_size[0] * self.kernel_size[0] * self.num_deformable_groups, ...]
        mask = offset_mask[:, 2 * self.kernel_size[0] * self.kernel_size[0] * self.num_deformable_groups: , ...]
        mask = torch.sigmoid(mask)
        ref_mask = 1 - mask
        
        if offset is None:
            B,_,H,W = ref_act.shape
            offset = torch.zeros((B, 2 * self.kernel_size[0] * self.kernel_size[0] * self.num_deformable_groups, H, W)).cuda()
        
        zero_offset = torch.zeros_like(offset)
        
        new_act = torchvision.ops.deform_conv2d(input=new_act, 
                                        offset=zero_offset, 
                                        weight=self.regular_conv.weight, 
                                        bias=self.regular_conv.bias, 
                                        padding=self.padding,
                                        stride=self.stride,
                                        mask=mask
                                        )
            

        
        if self.opt.isTrain:
            unmasked_ref_act = torchvision.ops.deform_conv2d(input=ref_act, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          stride=self.stride,
                                          mask=None
                                          )
        
        ref_act = torchvision.ops.deform_conv2d(input=ref_act, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          stride=self.stride,
                                          mask=ref_mask
                                          )
        
        
        
        
        
        output = ref_act + new_act
        
        
        return output, ref_act, new_act, unmasked_ref_act, mask