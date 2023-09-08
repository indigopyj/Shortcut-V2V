import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from models.modules.deform_conv import DeformableConv2d
from collections import OrderedDict
from models.modules.HetConv_module import HetConv
import torch.nn.functional as F
from models.offset_modules.offset_generator_local import LocalOffsetGenerator
from models.offset_modules.offset_generator_global import GlobalOffsetGenerator
from models.modules.deform_conv import DeformableConv2d
from models.modules.AdaBD_module import AdaBD

class ShortcutBlock(nn.Module):
    def __init__(self, opt, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(ShortcutBlock, self).__init__()
        self.opt=opt
        self.num_deformable_groups = self.opt.num_dg
        self.offset_k_size = 3
        self.offset_g_size = self.opt.offset_g_size
        self.offset_l_size = self.opt.offset_l_size
        local_offset_in_ch = self.opt.h_dim * 2
        local_offset_out_ch = 3 * self.offset_l_size * self.offset_l_size * self.num_deformable_groups
        
        self.offset_generator_global = GlobalOffsetGenerator(opt, input_nc, self.opt.h_dim, 2 * self.offset_g_size * self.offset_g_size * self.num_deformable_groups, n_layers=self.opt.g_layers)
        self.offset_generator_local = LocalOffsetGenerator(opt, local_offset_in_ch, self.opt.h_dim, local_offset_out_ch, n_layers=self.opt.l_layers)
        
        self.global_deform_conv = DeformableConv2d(self.opt.h_dim, self.opt.h_dim, kernel_size=3, num_deformable_groups=self.num_deformable_groups)
        
        self.cr1 = nn.Sequential(nn.Conv2d(input_nc//2, self.opt.h_dim, kernel_size=1), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(input_nc//2, self.opt.h_dim, kernel_size=1), nn.ReLU())
        
        self.mask_deform_conv = AdaBD(self.opt.h_dim, self.opt.h_dim, 3, self.num_deformable_groups, padding=self.opt.adabd_dilation, dilation=self.opt.adabd_dilation, opt=self.opt) # changed
        self.recon_ref = nn.Conv2d(self.opt.h_dim, output_nc, 1)
        
              
    def forward(self, a_1, a_2, f_1):
        new_act = None
        b, c, h,w = a_1.shape
        cr_f_1 = self.cr1(f_1)
        cr_a = self.cr2(torch.cat([a_1, a_2], dim=0))
        cr_a_1, cr_a_2 = cr_a[:b], cr_a[b:]
        # original ----------------------------------------
        # cr_a_2 = self.cr2(a_2)
        # cr_a_1 = self.cr2(a_1)
        # -------------------------------------------------
        ds_input = F.interpolate(torch.cat([a_1, a_2], dim=0), size=(h//2, w//2), mode='bilinear')
        a1_ds, a2_ds = ds_input[:b], ds_input[b:]
        global_input = F.interpolate(torch.cat([cr_a_1, cr_f_1], dim=0), size=(h//2, w//2), mode='bilinear')
        # original ---------------------------------------------------------
        # a1_d = F.interpolate(a_1, size=(h//2, w//2), mode='bilinear')
        # a2_d = F.interpolate(a_2, size=(h//2, w//2), mode='bilinear')
        # global_input = torch.cat((cr_a_1, cr_f_1), dim=0)
        # global_input = F.interpolate(global_input, size=(global_input.shape[2] // 2, global_input.shape[3] // 2), mode='bilinear')     
        # -------------------------------------------------------------------
        global_offset = self.offset_generator_global(a1_ds, a2_ds)
        global_offset_2 = global_offset.repeat(2,1,1,1)
        global_input = self.global_deform_conv(x=global_input, offset=global_offset_2)
        global_input = F.interpolate(global_input, size=(global_input.shape[2] * 2, global_input.shape[3] * 2), mode='bilinear')
        cr_a_1, cr_f_1 = global_input[:b], global_input[b:]
        offset_l_input = cr_a_1.detach()
        offset_mask = self.offset_generator_local(offset_l_input, cr_a_2)
        
        # Adaptive Blending and Deformation(AdaBD)
        new_ref, ref_act, new_act, unmasked_ref_act, mask = self.mask_deform_conv(ref_act=cr_f_1, new_act=cr_a_2, offset_mask=offset_mask)
        
        if self.opt.isTrain:
            ref_act = self.recon_ref(unmasked_ref_act)
        new_ref = self.recon_ref(new_ref)
        
        return new_ref, mask, ref_act