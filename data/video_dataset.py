import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch
from util.util import scale_img


class VideoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if opt.phase == "test" or opt.phase == "val":
            phase = "val"
        else:
            phase = "train"
        self.dir_A = os.path.join(opt.dataroot, 'Viper', phase, 'img')
        self.dir_B = os.path.join(opt.dataroot, "Cityscapes_sequence", "leftImg8bit", phase)
        
        if self.opt.dataset_option == 'v2c':
            self.dir_A = os.path.join(opt.dataroot, 'Viper', "recyclegan_" + phase, 'img')
            self.dir_B = os.path.join(opt.dataroot, "Cityscapes_sequence", "leftImg8bit", phase)   
        elif self.opt.dataset_option == 'v2l':
            self.dir_A = os.path.join(opt.dataroot, phase, "A")
            self.dir_B = os.path.join(opt.dataroot, phase, "B")
        elif self.opt.dataset_option == 'l2v':
            self.dir_A = os.path.join(opt.dataroot, phase, "B")
            self.dir_B = os.path.join(opt.dataroot, phase, "A")
            

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.seq_list = sorted(os.listdir(self.dir_A))
        
        if self.opt.dataset_option =='l2v' or self.opt.dataset_option == 'v2l':
            seq_list_tmp = []
            name = ''
            img_list = []
            for f in self.seq_list:
                if name != f[:3]:
                    if len(img_list) != 0:
                        seq_list_tmp.append(img_list)
                    img_list = [f]
                    name = f[:3]
                else:
                    img_list.append(f)
            
            seq_list_tmp.append(img_list)
            self.seq_list = seq_list_tmp
        
                    

        # self.transform = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        if self.opt.dataset_option =='l2v' or self.opt.dataset_option == 'v2l':
            seq_list = [f for f in self.seq_list[index]]
            A_path = seq_list
            seq_path = self.dir_A
            if self.opt.phase == "test":
                return {'seq_list' : seq_list, 'seq_path' : seq_path }
        else:
            seq_path = os.path.join(self.dir_A, self.seq_list[index])
        if 'ObamaTrump' in self.opt.dataroot or 'OliverColbert' in self.opt.dataroot:
            seq_path = self.dir_A
            
        if self.opt.phase == "test":
            return {'seq_path' : seq_path }
        
        if self.opt.dataset_option =='l2v' or self.opt.dataset_option == 'v2l':
            #A_path = seq_list[index]
            img_root = self.dir_A
        else:
            A_path = sorted([f for f in os.listdir(seq_path) if f.endswith(".jpg") or f.endswith(".png")])
            img_root = seq_path
        interval = torch.randint(1, self.opt.max_interval, [1]).item()
        if self.opt.phase != 'train':
            idx1 = torch.randint(0, len(A_path) - self.opt.max_interval, [1]).item()
        else:
            idx1 = torch.randint(0, len(A_path) - interval, [1]).item()
        
        
        img1 = Image.open(os.path.join(img_root, A_path[idx1])).convert("RGB") # change
        img2 = Image.open(os.path.join(img_root, A_path[idx1 + interval])).convert("RGB") #change

        # get the triplet from A
        img1 = scale_img(img1, self.opt, self.transform)
        img2 = scale_img(img2, self.opt, self.transform)
        
        img_path = A_path[idx1]

        return {'img1': img1, 'img2': img2, "img1_paths": A_path[idx1], "img_root": img_root}

    def __len__(self):
        return len(self.seq_list)

    def name(self):
        return 'VideoDataset'

