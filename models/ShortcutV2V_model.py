import copy
import os

import torch
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models import networks
from .base_model import BaseModel
from util.util import tensor2im, save_image
import cv2
import numpy as np
from torch.nn import functional as F
from torch import nn
from util.image_pool import ImagePool
from models.networks import init_weights
from util.util import warp, make_mask_heatmap
from pytorch_msssim import ssim
from thop import profile
import time
from models.modules.Shortcut_block import ShortcutBlock
from util.eval import compute_psnr
import math


def init_net(net, gpu_ids, init_type):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type)
    return net

def reset_parameters(module): # new
    n = module.in_channels
    for k in module.kernel_size:
        n *= k
    stdv = 1. / math.sqrt(n)
    module.weight.data.uniform_(-stdv, stdv)
    if module.bias is not None:
        module.bias.data.zero_()

def init_offset(m, num_dg, mask_initial):
    classname = m.__class__.__name__
    if classname.find('HetConv') != -1:
        return
    if classname.find('Conv') != -1:
        idx = m.kernel_size[0] * m.kernel_size[0] * num_dg        
        if m.weight.data.shape[0] < idx * 3:
            idx = 1
        
        m.weight.data.zero_()
        if m.bias is not None:
            if mask_initial != 0:
                m.bias.data[:-idx].zero_()
                m.bias.data[-idx:] = torch.tensor(mask_initial)    
                print("%s %d-initialize DCN offset" % (classname, mask_initial))
            else:
                m.bias.data.zero_()
    else:
        return
        
def init_Offsetgenerator(net, num_dg, mask_initial):
    for name, module in net.named_modules():
        if 'regular' in name:
            print("%s initialize DCN offset" % name)
            reset_parameters(module)
        if 'offset_conv' in name or 'offset_mask_conv' in name:
            print("%s zero-initialize DCN offset" % name)
            module.apply(lambda m: init_offset(m, num_dg, mask_initial))
    
    return net


def create_eval_dataloader(opt, phase="val"):
    opt = copy.deepcopy(opt)
    opt.isTrain = False
    opt.serial_batches = True
    # if opt.feat_vis:
    #     opt.serial_batches = False
    #     torch.manual_seed(567)
    opt.phase = phase # 고쳐야됨
    dataloader = CreateDataLoader(opt)
    dataloader = dataloader.load_data()
    return dataloader


class ShortcutV2VModel(BaseModel):
    def __init__(self, opt):
        super(ShortcutV2VModel, self).__init__()
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.loss_names = []
        self.loss_names += ['G_A', 'D_A']
        self.visual_names = ['img1', 'img2', 'fake_diff', 'real_diff']
        self.model_names = ['S']
        

        
        
        self.netS = ShortcutBlock(opt=self.opt, input_nc=self.opt.feat_ch*2, output_nc=self.opt.feat_ch)
        self.opt.init_type = 'xavier'
        print(self.netS)
        
        self.netS = init_net(self.netS, opt.gpu_ids, opt.init_type)
        
        if opt.isTrain:
            self.netS = init_Offsetgenerator(self.netS, self.opt.num_dg, self.opt.mask_initial)
            self.criterionGAN = networks.GANLoss_custom(gan_mode='lsgan').to(self.device)
            if self.opt.lambda_L1 > 0.0:
                self.loss_names += ['l1']
                self.criterionL1 = torch.nn.L1Loss()
            if self.opt.lambda_align > 0.0:
                self.loss_names += ['align']
                self.criterionL1 = torch.nn.L1Loss()
            if self.opt.lambda_L1_out > 0.0:
                self.loss_names += ['l1_out']
                self.criterionL1_out = torch.nn.L1Loss()
            if opt.lambda_lpips > 0:
                self.loss_names += ['lpips']
                from models import lpips
                self.netLPIPS = lpips.PerceptualLoss(model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=True,)
    
                        
            d_input_dim = 3
            ndf_d = 64
            self.netD_A = networks.define_D(d_input_dim, ndf_d,
                                    "sn_basic",
                                    opt.Shortcut_n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids)
            
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            self.netD_T_A = networks.define_D(d_input_dim, ndf_d,
                                        "temporal_patchGAN",
                                        opt.Shortcut_n_layers_D, opt.norm, False, opt.init_type, self.gpu_ids) # temporal discriminator
            self.loss_names += ['D_T', 'G_T']
            
            if opt.continue_train:
                self.load_network(self.netS, 'ShortcutV2V', self.opt.which_epoch)
                self.load_network(self.netD_A, 'D_A', self.opt.which_epoch)  
                if self.opt.Temporal_GAN_loss:
                    self.load_network(self.netD_T_A, 'D_T_A', self.opt.which_epoch) 
                
            self.optimizer = torch.optim.Adam(self.netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer)
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D_A)
            
            self.criterionGAN = networks.GANLoss_custom(gan_mode='lsgan').to(self.device)
            self.optimizer_D_T_A = torch.optim.Adam(self.netD_T_A.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D_T_A)
            self.loss_D_T = 0.0

            self.eval_dataloader = create_eval_dataloader(self.opt, "val")
        

    def setup_with_G(self, opt, model, verbose=True):
        if len(self.opt.gpu_ids) > 1:
            self.modelG = model.netG_A.module
        else:
            self.modelG = model.netG_A
        if not opt.isTrain or opt.continue_train:
            print("load %s network" % opt.which_epoch)
            self.load_network(self.netS, "ShortcutV2V", opt.which_epoch)
        if self.opt.main_G_path != None:
            self.load_pretrained_network(self.modelG, self.opt.main_G_path)
        if self.opt.phase == "train":
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        for param in self.modelG.parameters():
            param.requires_grad = False
        print("freeze main generator")
    

    def set_input(self, input):
        self.ref_img = input['img1'].to(self.device)
        self.curr_img = input['img2'].to(self.device)
        self.ref_img_paths = input['img1_paths']
        self.image_root = input['img_root']
    
    def set_test_input(self, img_paths): # load an image from img_paths and preprocess it as a train input.
        from PIL import Image
        import torchvision.transforms as transforms
        from util.util import scale_img
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        
        self.transform = transforms.Compose(transform_list)
        self.curr_img_paths = img_paths
        self.curr_img = []
        for i in range(len(img_paths)):
            img = Image.open(img_paths[i]).convert("RGB")
            img = scale_img(img, self.opt, self.transform).unsqueeze(0)
            self.curr_img.append(img)
        self.curr_img = torch.cat(self.curr_img, dim=0).to(self.device)

    def forward(self):
        b = self.ref_img.size(0)
        activations1 = self.modelG.model[:self.opt.skip_idx_start](torch.cat((self.ref_img, self.curr_img), 0)).detach()
        activations2 = self.modelG.model[self.opt.skip_idx_start:self.opt.skip_idx_end](activations1).detach()
        self.ref_feat_a, self.curr_feat_a = activations1[:b], activations1[b:]

        self.ref_feat_f = activations2[:b] # ref_img
        self.curr_feat_f = activations2[b:] # curr_img

        self.fake_feat, self.mask, self.warped_ref_feat = self.netS(self.ref_feat_a, self.curr_feat_a, self.ref_feat_f)
        
        self.ref_im = self.modelG.model[self.opt.skip_idx_end:](self.ref_feat_f)
    
        self.real_im = self.modelG.model[self.opt.skip_idx_end:](self.curr_feat_f).detach()
        self.fake_im = self.modelG.model[self.opt.skip_idx_end:](self.fake_feat)
        

    def backward(self):
        self.loss = 0.0
        self.loss_l1 = self.criterionL1(self.fake_feat, self.curr_feat_f)
        self.loss += self.loss_l1 * self.opt.lambda_L1

        self.loss_align = self.criterionL1(self.warped_ref_feat, self.curr_feat_f)
        self.loss += self.loss_align * self.opt.lambda_align
        
        if self.opt.lambda_L1_out > 0.0:
            self.loss_l1_out = self.criterionL1_out(self.fake_im, self.real_im)
            self.loss += self.loss_l1_out * self.opt.lambda_L1_out
        else:
            self.loss_l1_out = 0.0
            
        # GAN loss
        pred_fake = self.netD_A(self.fake_im)
        self.loss_G_A = self.criterionGAN(pred_fake, True, for_discriminator=False)
        self.loss += self.loss_G_A
        
        fake_B = torch.stack((self.ref_im, self.fake_im), 2)
        pred_fake = self.netD_T_A(fake_B)
        self.loss_G_T = self.opt.lambda_D_T * self.criterionGAN(pred_fake, False)
        self.loss += self.loss_G_T

        if self.opt.lambda_lpips > 0:
            self.loss_lpips = self.netLPIPS(self.fake_im, self.real_im).mean()
            self.loss += self.opt.lambda_lpips * self.loss_lpips
        
        self.loss.backward()

    
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_im)
        real = self.real_im
        self.loss_D_A = self.backward_D_basic(self.netD_A, real, fake_B)
 
    
    def calc_D_T_loss(self, past_frame, real_frame, fake_frame):
        # shape: B, C, L, H, W
        real_B = torch.stack((past_frame, real_frame), 2)
        fake_B = torch.stack((past_frame, fake_frame), 2)
        pred_real = self.netD_T_A(real_B)
        loss_D_T_real = self.criterionGAN(pred_real, True)
        
        pred_fake = self.netD_T_A(fake_B.detach())
        loss_D_T_fake = self.criterionGAN(pred_fake, False)
        loss_D_T = (loss_D_T_real + loss_D_T_fake) * 0.5
        return loss_D_T

    def backward_D_T_A(self):
        self.loss_D_T = self.opt.lambda_D_T * self.calc_D_T_loss(self.ref_im, self.real_im, self.fake_im)
        self.loss_D_T.backward()
        
    def optimize_parameters(self, epoch):
            self.forward()
            self.optimizer.zero_grad()
            self.backward()
            self.optimizer.step()
            
            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()
            
            if self.opt.Temporal_GAN_loss:
                self.optimizer_D_T_A.zero_grad()
                self.backward_D_T_A()
                self.optimizer_D_T_A.step()
        
    def save(self, label):
        self.save_network(self.netS, 'ShortcutV2V', label, self.gpu_ids)
        self.save_network(self.netD_A, "D_A", label, self.gpu_ids)
        self.save_network(self.netD_T_A, "D_T_A", label, self.gpu_ids)


    def evaluate_model(self, step):
        self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        save_dir = os.path.join(self.save_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netS.eval()
        self.opt.isTrain = False
        
        with torch.no_grad(): 
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
                if i >= 10:  # manually limit the num of validation videos
                    break
                self.set_input(data_i) # load random video and random index image
                ref_feat_a = self.modelG.model[:self.opt.skip_idx_start](self.ref_img).detach()
                ref_feat_f = self.modelG.model[self.opt.skip_idx_start:self.opt.skip_idx_end](ref_feat_a)
                
                for j in range(1, self.opt.max_interval):
                    img2_paths = []
                    for batch_idx in range(len(self.ref_img_paths)):
                        img1_name, img1_ext = os.path.splitext(self.ref_img_paths[batch_idx])
                        img2_idx = int(img1_name.split("_")[1]) + j
                        img2_name =  "%s_%05d%s" %(img1_name.split("_")[0], img2_idx, img1_ext)
                        img2_path = os.path.join(self.image_root[batch_idx], img2_name) 
                        img2_paths.append(img2_path)
                self.set_test_input(img2_paths) # load an image (img1 + interval) as a batch : self.curr_img
                    
                curr_feat_a = self.modelG.model[:self.opt.skip_idx_start](self.curr_img).detach()
                curr_feat_f = self.modelG.model[self.opt.skip_idx_start:self.opt.skip_idx_end](curr_feat_a)
                
                fake_feat, _, _, = self.netS(ref_feat_a, curr_feat_a, ref_feat_f)

                real_im = self.modelG.model[self.opt.skip_idx_end:](curr_feat_f)
                fake_im = self.modelG.model[self.opt.skip_idx_end:](fake_feat)
        
                for k in range(len(self.ref_img_paths)):
                    img1_name, ext = os.path.splitext(self.ref_img_paths[k])
                    name = f"{img1_name}_{j}{ext}" # interval_originalname
                    input1_im = tensor2im(self.ref_img, idx=k)
                    input2_im = tensor2im(self.curr_img, idx=k)
                    real = tensor2im(real_im, idx=k)
                    fake = tensor2im(fake_im, idx=k)
                    save_image(input1_im, os.path.join(save_dir, 'input1', '%s' % self.ref_img_paths[k]), create_dir=True)
                    save_image(input2_im, os.path.join(save_dir, 'input2', '%s' % name), create_dir=True)
                    save_image(real, os.path.join(save_dir, 'real', '%s' % name), create_dir=True)
                    save_image(fake, os.path.join(save_dir, 'fake', '%s' % name), create_dir=True)
                                

        self.netS.train()
        self.opt.isTrain = True
    
    def test_model(self, result_path):
        os.makedirs(result_path, exist_ok=True)
        
        self.netS.eval()
        self.test_dataloader = create_eval_dataloader(self.opt, "test")
        total_spent = 0
        count = 0
        

        ssim_total = 0
        psnr_total = 0
        red_count = 0
        
        img_num = 3

        if self.opt.save_image: self.opt.how_many = 10000000
        with torch.no_grad():
            for seq_idx, seq_i in enumerate(tqdm(self.test_dataloader, desc='Eval       ', position=2, leave=False)):
                if seq_idx >= self.opt.how_many:  # only apply our model to opt.how_many videos.
                    break
                if self.opt.dataset_option == 'v2l' or self.opt.dataset_option == 'l2v':
                    vid_name = seq_i['seq_list'][0][0][:3]
                    img_list = sorted([f[0] for f in seq_i['seq_list']])
                else:
                    vid_name = seq_i['seq_path'][0].split("/")[-1]
                    img_list = sorted(os.listdir(seq_i['seq_path'][0]))
                
                if not self.opt.save_image:
                    video = cv2.VideoWriter(os.path.join(result_path, vid_name+'.mp4'), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=8, frameSize=(self.opt.fineSizeW*img_num, self.opt.fineSizeH))
                    
                if not self.opt.save_image: # only save 200 frame video for fast evaluation
                    img_list = img_list[:200]
                print(f"{seq_i['seq_path']} : {len(img_list)}")
                fake_feat = curr_feat_f = None
                for i, data_i in enumerate(img_list):
                    data_path = os.path.join(seq_i['seq_path'][0], data_i)
                    data_path = [data_path]
                    self.set_test_input(data_path)
                    B = self.curr_img.shape[0]
                    if i % self.opt.max_interval == 0:
                        reference_img = self.curr_img
                        if i == 0:
                            ref_feat_a = self.modelG.model[:self.opt.skip_idx_start](reference_img).detach()
                            ref_feat_f = self.modelG.model[self.opt.skip_idx_start:self.opt.skip_idx_end](ref_feat_a)
                            fake_im = self.modelG.model[self.opt.skip_idx_end:](ref_feat_f)
                            real_im = fake_im
                        else:
                            # generate output frame using old reference features but update the reference features
                            curr_feat_a = self.modelG.model[:self.opt.skip_idx_start](reference_img).detach() # a_t
                            curr_feat_f = self.modelG.model[self.opt.skip_idx_start:self.opt.skip_idx_end](curr_feat_a) # f_t
                            fake_feat, _, _ = self.netS(ref_feat_a, curr_feat_a, ref_feat_f) # S(a_ref, a_t, f_ref)
                            ref_feat_a = curr_feat_a.clone()
                            ref_feat_f = curr_feat_f.clone()
                            
                        count += 1
                        
                    else:
                        curr_feat_a = self.modelG.model[:self.opt.skip_idx_start](self.curr_img).detach()
                        curr_feat_f = self.modelG.model[self.opt.skip_idx_start:self.opt.skip_idx_end](curr_feat_a)
                        fake_feat, _, _ = self.netS(ref_feat_a, curr_feat_a, ref_feat_f)
                        count += 1
                        red_count += 1
                    
                    if fake_feat is not None and curr_feat_f is not None:
                        output = self.modelG.model[self.opt.skip_idx_end:](torch.cat([fake_feat, curr_feat_f], dim=0))
                        fake_im, real_im = output[:B], output[B:]
                    
                    if i % self.opt.max_interval != 0:
                        ssim_value = ssim(fake_im, real_im, data_range=1, size_average=True)
                        ssim_total += ssim_value.item()
                        psnr_value = compute_psnr(fake_im, real_im, max_value = 1)
                        psnr_total += psnr_value.item()
                    
                    img_name = i+1
                    name = f"{vid_name}_{img_name:05d}.png"
                    input_im = tensor2im(self.curr_img)
                    real_im = tensor2im(real_im)
                    fake_im = tensor2im(fake_im)
                    
                    if self.opt.save_image:
                        save_image(fake_im, os.path.join(result_path, 'fake', vid_name, '%s' % name), create_dir=True)
                    cat_img = np.concatenate((input_im, real_im,  fake_im), axis=1)
                        
                    if not self.opt.save_image:
                        cat_img = cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR)
                        video.write(cat_img)

            print(f'Spent time : ', total_spent / count)
            macs_org, params_org = profile(self.modelG, (reference_img,))
            print('Original MACs: %.3fG   Original Params: %.3fM' % (macs_org / 1e9, params_org / 1e6))
            input_shape = (ref_feat_a, curr_feat_a, ref_feat_f,)
            macs_red, params_red = profile(self.netS, input_shape)
            macs, params = profile(self.modelG.model[self.opt.skip_idx_end:], (fake_feat,))
            macs += macs_red 
            params += params_red

            macs_back, params_back = profile(self.modelG.model[:self.opt.skip_idx_start], (self.curr_img,))
            macs += macs_back
            params += params_back
                
            print('Total MACs: %.3fG' % (macs / 1e9))
            print('RED MACs: %.3fG' % (macs_red / 1e9))
            print('Reduced percentage: ', 1- macs/macs_org)
            
            print('Total Params: %.3fM' % (params / 1e6))
            print('RED Params: %.3fM' % (params_red / 1e6))
            print('Reduced percentage: ', 1- params/params_org)
            
            print('AVG SSIM: ' ,ssim_total/red_count)
            print('AVG PSNR: ', psnr_total/red_count)
    
    
    def get_current_errors(self):
        from collections import OrderedDict
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
