import os
import torch
from torch import nn

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if len(gpu_ids) and torch.cuda.is_available():
            if isinstance(network, nn.DataParallel):
                    torch.save(network.module.cpu().state_dict(), save_path)
            else:
                torch.save(network.cpu().state_dict(), save_path)
            network.cuda(gpu_ids[0])
        else:
            torch.save(network.cpu().state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        weight = torch.load(save_path)
        if isinstance(network, nn.DataParallel):
            network = network.module
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            if "decoder" in k: # bug
                continue
            if k[:6] == "module":
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        
        network.load_state_dict(new_state_dict, strict=True)
        
        
    def load_pretrained_network(self, network, network_path):
        weight = torch.load(network_path)
        if isinstance(network, nn.DataParallel):
            network = network.module
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            if "decoder" in k: # bug
                continue
            if k[:6] == "module":
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
            
        network.load_state_dict(new_state_dict, strict=True)
        #network.load_state_dict(torch.load(network_path), strict=False)
    
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
