from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--which_iter', type=int, default=0, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--d_lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--identity', type=float, default=0.5, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--adversarial_loss_p', action='store_true', help='also train the prediction model with an adversarial loss')

        self.parser.add_argument('--lambda_cycle_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_cycle_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_unsup_cycle_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_unsup_cycle_B', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.0, help='')
        self.parser.add_argument('--lambda_spa_unsup_A', type=float, default=10.0, help='')
        self.parser.add_argument('--lambda_spa_unsup_B', type=float, default=10.0, help='')
        self.parser.add_argument('--lambda_content_A', type=float, default=0.0, help='')
        self.parser.add_argument('--lambda_content_B', type=float, default=0.0, help='')
        self.parser.add_argument('--motion_level', type=float, default=8., help='weight')
        self.parser.add_argument('--scale_level', type=float, default=0, help='weight')
        self.parser.add_argument('--shift_level', type=float, default=10., help='weight')
        self.parser.add_argument('--noise_level', type=float, default=0.001, help='weight')
        
        # for ShortcutV2V
        self.parser.add_argument("--lambda_L1", type=float, default=5.0, help='lambda for L1 loss')
        self.parser.add_argument("--lambda_L1_out", type=float, default=10)
        self.parser.add_argument("--lambda_lpips", type=float, default=10)
        self.parser.add_argument("--lambda_D_T", type=float, default=1.0)
        self.parser.add_argument("--lambda_align", type=float, default=5.0)
        self.parser.add_argument("--spectral_norm", type=bool, default=True)
        self.parser.add_argument("--Temporal_GAN_loss", type=bool, default=True)
        self.parser.add_argument('--eval_freq', type=int, default=100, help='frequency of showing evaluation results and save them in local')
        self.parser.add_argument("--mask_initial", type=float, default=0.0)
        

        self.isTrain = True
        self.Test_RED = False
