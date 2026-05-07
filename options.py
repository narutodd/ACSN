import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--dataroot_test', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize', action='store_true', help='weather to resize the picture to (256,256)')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--input_dim_a', type=int, default=1, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=1, help='# of input channels for domain B')
    self.parser.add_argument('--num_classes', type=int, default=2, help='# of input channels for domain B')
    self.parser.add_argument('--nThreads', type=int, default=0, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='autoPET_ACSN', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_seg', type=float, default=1e-4, help='segmentation lr')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=200, help='number of epochs')
    self.parser.add_argument('--n_ep_decay', type=int, default=0, help='epoch start decay learning rate, set -1 if no decay')
    self.parser.add_argument('--lambda_epochs', type=int, default=50, metavar='N', help='gradually increase the value of beta from 0 to 1 when training')
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--early_stop', type=int, default=20, help='Weather to early stop')
    self.parser.add_argument('--lambda_L1', type=int, default=100, help='hyperparameter lambda2')
    self.parser.add_argument('--lambda_seg', type=int, default=1, help='hyperparameter lambda4')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--n_downsampling', type=int, default=2, help='number of down sample in Generator')
    self.parser.add_argument('--nz', type=int, default=8, help='size of style feature')
    self.parser.add_argument('--isTrain', type=bool, default=True)

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot_test', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--resize', action='store_true', help='weather to resize the picture to (256,256)')
    self.parser.add_argument('--input_dim_a', type=int, default=1, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=1, help='# of input channels for domain B')
    self.parser.add_argument('--num_classes', type=int, default=2, help='# of input channels for domain B')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='autoPET_ACSN', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')

    # model related

    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--n_downsampling', type=int, default=2, help='number of down sample in Generator')
    self.parser.add_argument('--nz', type=int, default=8, help='size of style feature')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
      self.opt.dis_scale = 3
      self.opt.dis_norm = 'None'
      self.opt.dis_spectral_norm = False
    return self.opt
