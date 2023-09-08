from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import torch.nn.functional as F
import random
import cv2
from torch.autograd import Variable
import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_offsets(result_path, img1_name, img2_name, img1, img2, point_list, offset_g, offset_l):
    s_img1 = img1.copy()
    s_img2 = img2.copy()
    g_img1 = img1.copy()
    g_img2 = img2.copy()
    l_img1 = img1.copy()
    l_img2 = img2.copy()
    
    offset_sum = offset_g + offset_l
    
    for i, (y,x) in enumerate(point_list):
    
        if i == 0:
            sum_plot = plot_offsets(s_img1, s_img2, offset_sum, y, x, True)
            global_plot = plot_offsets(g_img1, g_img2, offset_g, y, x, True)
            local_plot = plot_offsets(l_img1, l_img2, offset_l, y, x, True)

        else:
            sum_plot = plot_offsets(sum_plot[0], sum_plot[1], offset_sum, y, x, True)
            global_plot = plot_offsets(global_plot[0], global_plot[1], offset_g, y, x, True)
            local_plot = plot_offsets(local_plot[0], local_plot[1], offset_l, y, x, True)
            
        cv2.imwrite(os.path.join(result_path, img2_name + ".png"), sum_plot[1])       
        cv2.imwrite(os.path.join(result_path, img1_name + "_sum.png"), sum_plot[0])
        cv2.imwrite(os.path.join(result_path, img1_name + "_global.png"), global_plot[0])
        cv2.imwrite(os.path.join(result_path, img1_name + "_local.png"), local_plot[0])

                


def plot_offsets(img1, img2, offsets, roi_x, roi_y, save=True):
    img1 = img1.copy()
    img2 = img2.copy()

    input_img_h, input_img_w = img1.shape[:2]
    #for offsets in save_output.outputs:
    offset_tensor_h, offset_tensor_w = offsets.shape[2:]
    resize_factor_h, resize_factor_w = input_img_h/offset_tensor_h, input_img_w/offset_tensor_w

    offsets_y = offsets[:, ::2]
    offsets_x = offsets[:, 1::2]

    grid_y = np.arange(0, offset_tensor_h)
    grid_x = np.arange(0, offset_tensor_w)

    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    sampling_y = grid_y + offsets_y.detach().cpu().numpy()
    sampling_x = grid_x + offsets_x.detach().cpu().numpy()

    sampling_y *= resize_factor_h
    sampling_x *= resize_factor_w

    sampling_y = sampling_y[0] # remove batch axis
    sampling_x = sampling_x[0] # remove batch axis

    sampling_y = sampling_y.transpose(1, 2, 0) # c, h, w -> h, w, c
    sampling_x = sampling_x.transpose(1, 2, 0) # c, h, w -> h, w, c

    sampling_y = np.clip(sampling_y, 0, input_img_h)
    sampling_x = np.clip(sampling_x, 0, input_img_w)

    sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
    sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

    sampling_y = sampling_y[roi_y, roi_x]
    sampling_x = sampling_x[roi_y, roi_x]
    
    for y, x in zip(sampling_y, sampling_x):
        y = round(y)
        x = round(x)
        cv2.circle(img1, center=(x, y), color=(0, 0, 255), radius=2, thickness=-1)
    
    cv2.circle(img1, center=(roi_x, roi_y), color=(0, 255, 0), radius=2, thickness=-1)
    cv2.circle(img2, center=(roi_x, roi_y), color=(0, 255, 0), radius=2, thickness=-1)
    return (img1, img2)

def correlate(input1, input2):
    from spatial_correlation_sampler import spatial_correlation_sample
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)

def channel_visualize(feat, ch=0, name='f', mul=1.0):
    import matplotlib.pyplot as plt

    f1 = copy.deepcopy(feat[0, ch, ...])
    #print(f1.shape)
    f1 = f1[14:114, 78:178]
    f1 -= f1.min()
    f1 /= f1.max()
    f1 *= mul
    f1_img = f1.cpu().float().numpy()
    cm = plt.get_cmap('Greys')
    color_img = (cm(f1_img)[:,:,:3]*255.0).astype(np.uint8)
    save_image(color_img, f"./channel_vis_patch_v2/{name}_{ch}.png")
    #f1_img = (np.transpose(f1.repeat(1,3,1,1)[0].cpu().float().numpy(), (1,2,0)) * 255.0).astype(np.uint8)
    #save_image(f1_img, f"./channel_vis/{name}_{ch}.png")


def make_mask_heatmap(activation, to_tensor=True, size=None, normalize=False):
    a = activation.cpu().numpy()
    output = np.abs(a)
    output = np.mean(output,axis=0).squeeze()
    if normalize:
        output -= output.min()
        output /= output.max()
    output *= 255
    output = output.astype('uint8')
    #output = 255 - output.astype('uint8')
    #heatmap = cv2.applyColorMap(output, cv2.COLORMAP_INFERNO)
    heatmap = cv2.applyColorMap(output, cv2.COLORMAP_HOT)
    if not size:
        heatmap = cv2.resize(heatmap, (512, 512))
    else:
        heatmap = cv2.resize(heatmap, size)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # if to_tensor:
    #     heatmap = transforms.ToTensor()(heatmap)
    return heatmap

def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid - flo
    
    # Scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear')
    return output

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array, from [-1, 1] to [0, 255], with the shape from (c, h, w) to (h, w, c)
def tensor2im(image_tensor, idx=0, imtype=np.uint8):
    image_numpy = image_tensor[idx].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
   # assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)

    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def scale_img(img, opt, transform):
    if opt.phase == "test" or opt.phase == "val":
        opt.loadSizeW = opt.fineSizeW
        opt.loadSizeH = opt.fineSizeH
    if opt.resize_mode == "scale_shortest":
        w, h = img.size
        if w >= h: 
            scale = opt.loadSize / h
            new_w = int(w * scale)
            new_h = opt.loadSize
        else:
            scale = opt.loadSize / w
            new_w = opt.loadSize
            new_h = int(h * scale)
            
        img = img.resize((new_w, new_h), Image.BICUBIC)
    elif opt.resize_mode == "square":
        img = img.resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
    elif opt.resize_mode == "rectangle":
        img = img.resize((opt.loadSizeW, opt.loadSizeH), Image.BICUBIC)
    elif opt.resize_mode == "none":
        pass
    else:
        raise ValueError("Invalid resize mode!")

    img = transform(img)
    
    if opt.phase == "test" or opt.phase == "val": # no random crop
        return img
    

    w = img.size(2)
    h = img.size(1)
    if opt.crop_mode == "square":
        fineSizeW, fineSizeH = opt.fineSize, opt.fineSize
    elif opt.crop_mode == "rectangle":
        fineSizeW, fineSizeH = opt.fineSizeW, opt.fineSizeH
    elif opt.crop_mode == "none":
        fineSizeW, fineSizeH = w, h
    else:
        raise ValueError("Invalid crop mode!")

    if 'ObamaTrump' in opt.dataroot or 'OliverColbert' in opt.dataroot:
        if opt.phase == "train":
            h_offset, w_offset = 0, 256 * random.randint(0, 2)
        else:
            h_offset, w_offset = 0, 0
    else:
        w_offset = random.randint(0, max(0, w - fineSizeW - 1))
        h_offset = random.randint(0, max(0, h - fineSizeH - 1))

    img = img[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]
    
    return img

