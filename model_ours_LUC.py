import networks_ours_LUC as networks
import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions import dce_evidence_u_loss
from networks_ours_LUC import DS_Combin


class ACSN(nn.Module):
    def __init__(self, opts):
        super(ACSN, self).__init__()

        # parameters
        if opts.phase == 'train':
            lr = 0.0001
            lr_dcontent = lr / 2.5
            lr_seg = opts.lr_seg
        self.nz = 8
        self.concat = opts.concat
        self.no_ms = opts.no_ms
        self.opts = opts

        # discriminators
        if opts.dis_scale > 1:
            self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm,
                                               sn=opts.dis_spectral_norm)
            self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm,
                                               sn=opts.dis_spectral_norm)
        else:
            self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.disContent = networks.Dis_content()

        # encoders
        self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)
        if self.concat:
            self.enc_a = networks.E_attr_concat(opts.input_dim_a, opts.input_dim_b, self.nz, \
                                                norm_layer=None,
                                                nl_layer=networks.get_non_linearity(layer_type='lrelu'))
        else:
            self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

        # generator
        if self.concat:
            self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
        else:
            self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

        # segment
        self.seg = networks.Seg(opts.num_classes)

        # optimizers
        if opts.phase == 'train':
            self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
            self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
            self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
            self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
            self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
            self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
            self.seg_opt = torch.optim.Adam(self.seg.parameters(), lr=lr_seg, betas=(0.5, 0.999), weight_decay=0.0001)

            # Setup the loss function for training
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSeg = networks.BCEDiceLoss()
            self.criterionSim = networks.cosine_similarity_loss()
            self.criterionMSE = torch.nn.MSELoss(reduction='mean')

    def initialize(self):
        self.disA.apply(networks.gaussian_weights_init)
        self.disB.apply(networks.gaussian_weights_init)
        self.disContent.apply(networks.gaussian_weights_init)
        self.gen.apply(networks.gaussian_weights_init)
        self.enc_c.apply(networks.gaussian_weights_init)
        self.enc_a.apply(networks.gaussian_weights_init)
        self.seg.apply(networks.gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
        self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
        self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
        self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)
        self.seg_sch = networks.get_scheduler(self.seg_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        self.disContent.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_a.cuda(self.gpu)
        self.gen.cuda(self.gpu)
        self.seg.cuda(self.gpu)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z

    def test_forward_transfer(self, image_a, image_b):
        z_content_a, z_content_b, content_features_a, content_features_b = self.enc_c.forward(image_a, image_b)
        if self.concat:
            mu_a, logvar_a, mu_b, logvar_b = self.enc_a.forward(image_a, image_b)
            std_a = logvar_a.mul(0.5).exp_()
            eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            z_attr_a = eps.mul(std_a).add_(mu_a)
            std_b = logvar_b.mul(0.5).exp_()
            eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            z_attr_b = eps.mul(std_b).add_(mu_b)
        else:
            z_attr_a, z_attr_b = self.enc_a.forward(image_a, image_b)
        fake_B = self.gen.forward_b(z_content_a, z_attr_b)
        fake_A = self.gen.forward_a(z_content_b, z_attr_a)

        output_fusion, output_A, output_B, u_a2, u_a1, u_a0, u_b2, u_b1, u_b0 = self.seg.forward(z_content_a, z_content_b, content_features_a, content_features_b, image_a, image_b)
        evidence_A = F.softplus(output_A)
        evidence_B = F.softplus(output_B)
        evidence_fusion = F.softplus(output_fusion)
        alpha_A = evidence_A.permute(0, 2, 3, 1).reshape(-1, self.opts.num_classes) + 1
        alpha_B = evidence_B.permute(0, 2, 3, 1).reshape(-1, self.opts.num_classes) + 1
        alpha_fusion = evidence_fusion.permute(0, 2, 3, 1).reshape(-1, self.opts.num_classes) + 1

        alpha = dict()
        alpha[0] = alpha_A
        alpha[1] = alpha_B
        alpha[2] = alpha_fusion
        alpha_last, u_last, u_A, u_B, u_fusion = DS_Combin(alpha, self.opts.num_classes)
        evidence_last = (alpha_last - 1).reshape(-1, self.opts.resize_size, self.opts.resize_size, self.opts.num_classes).permute(0, 3, 1, 2)

        uncertainty_last = u_last.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)
        uncertainty_A = u_A.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)
        uncertainty_B = u_B.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)
        uncertainty_fusion = u_fusion.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)

        predict_A = ((evidence_A+1) / torch.sum(evidence_A + 1, dim=1, keepdim=True))
        predict_B = ((evidence_B+1) / torch.sum(evidence_B + 1, dim=1, keepdim=True))
        predict_fusion = ((evidence_fusion+1) / torch.sum(evidence_fusion + 1, dim=1, keepdim=True))
        predict = ((evidence_last+1) / torch.sum(evidence_last + 1, dim=1, keepdim=True))
        return fake_A, fake_B, predict, u_a2, u_a1, u_a0, u_b2, u_b1, u_b0, uncertainty_A, uncertainty_B, z_content_a, z_content_b

    def forward(self):
        self.real_A_encoded = self.input_A.detach()
        self.real_B_encoded = self.input_B.detach()
        self.mask_encoded = self.mask.detach()

        # get encoded z_c
        self.z_content_a, self.z_content_b, _, _ = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

        # get encoded z_a
        if self.concat:
            self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
            std_a = self.logvar_a.mul(0.5).exp_()
            eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
            std_b = self.logvar_b.mul(0.5).exp_()
            eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
        else:
            self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)

        # first cross translation
        input_content_forA = torch.cat((self.z_content_b, self.z_content_a), 0)
        input_content_forB = torch.cat((self.z_content_a, self.z_content_b), 0)
        input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a), 0)
        input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b), 0)
        output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
        output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
        self.fake_A_encoded, self.fake_AA_encoded = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
        self.fake_B_encoded, self.fake_BB_encoded = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

        # for display
        self.image_display = torch.cat((
            self.real_A_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(),
            self.fake_AA_encoded[0:1].detach().cpu(),
            self.real_B_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
            self.fake_BB_encoded[0:1].detach().cpu()), dim=0)

    def forward_content(self):
        half_size = 1
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]
        # get encoded z_c
        self.z_content_a, self.z_content_b, _, _ = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

    def update_D_content(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward_content()
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self, image_a, image_b, mask):
        self.input_A = image_a
        self.input_B = image_b
        self.mask = mask
        self.forward()
        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded, self.real_B_encoded)
        self.disA_loss = loss_D1_A.item()
        self.disA_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded, self.real_A_encoded)
        self.disB_loss = loss_D1_B.item()
        self.disB_opt.step()

        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()
        self.D_loss = self.disContent_loss + self.disA_loss + self.disB_loss

    def backward_D(self, netD, real, fake, condition):
        pred_fake = netD.forward(fake.detach(), condition)
        pred_real = netD.forward(real, condition)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            # out_fake = torch.sigmoid(out_a)
            # out_real = torch.sigmoid(out_b)
            out_fake = out_a
            out_real = out_b
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            # ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            # ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)

            ad_fake_loss = self.criterionMSE(out_fake, all0)
            ad_true_loss = self.criterionMSE(out_real, all1)
            loss_D = loss_D + ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def backward_contentD(self, imageA, imageB):
        pred_fake = self.disContent.forward(imageA.detach())
        pred_real = self.disContent.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            # out_fake = torch.sigmoid(out_a)
            # out_real = torch.sigmoid(out_b)
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            # ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            # ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_fake_loss = self.criterionMSE(out_fake, all0)
            ad_true_loss = self.criterionMSE(out_real, all1)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_EG(self):
        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()

    def backward_EG(self):
        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA, self.real_B_encoded)
        loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB, self.real_A_encoded)

        # KL loss - z_a
        if self.concat:
            kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
            loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
            kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
            loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
        else:
            loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
            loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
        loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01

        # cross cycle consistency loss
        loss_G_L1_A = self.criterionL1(self.fake_A_encoded, self.real_A_encoded) * self.opts.lambda_L1
        loss_G_L1_B = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opts.lambda_L1
        loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

        loss_G = loss_G_GAN_A + loss_G_GAN_B + loss_G_GAN_Acontent + loss_G_GAN_Bcontent + loss_G_L1_AA + loss_G_L1_BB + loss_G_L1_A + loss_G_L1_B + loss_kl_zc_a + loss_kl_zc_b + loss_kl_za_a + loss_kl_za_b
        loss_G.backward()

        self.G_loss = loss_G.item()

    def backward_G_GAN_content(self, data):
        outs = self.disContent.forward(data)
        for out in outs:
            # outputs_fake = torch.sigmoid(out)
            outputs_fake = out
            all_half = 0.5 * torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            # ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
            ad_loss = self.criterionMSE(outputs_fake, all_half)
        return ad_loss

    def backward_G_GAN(self, fake, netD, condition):
        outs_fake = netD.forward(fake, condition)
        loss_G = 0
        for out_a in outs_fake:
            # outputs_fake = torch.sigmoid(out_a)
            outputs_fake = out_a
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            # loss_G = loss_G + nn.functional.binary_cross_entropy(outputs_fake, all_ones)
            loss_G = loss_G + self.criterionMSE(outputs_fake, all_ones)
        return loss_G

    def forword_seg(self):
        z_contents_a, z_contents_b, content_features_a, content_features_b = self.enc_c.forward(self.images_A, self.images_B)

        output_fusion, output_A, output_B = self.seg.forward(z_contents_a.detach(), z_contents_b.detach(), content_features_a, content_features_b, self.images_A, self.images_B)

        self.evidence_A = F.softplus(output_A)
        self.evidence_B = F.softplus(output_B)
        self.evidence_fusion = F.softplus(output_fusion)
        self.alpha_A = self.evidence_A.permute(0, 2, 3, 1).reshape(-1, self.opts.num_classes) + 1
        self.alpha_B = self.evidence_B.permute(0, 2, 3, 1).reshape(-1, self.opts.num_classes) + 1
        self.alpha_fusion = self.evidence_fusion.permute(0, 2, 3, 1).reshape(-1, self.opts.num_classes) + 1

        alpha = dict()
        alpha[0] = self.alpha_A
        alpha[1] = self.alpha_B
        alpha[2] = self.alpha_fusion
        self.alpha_last, u_last, u_A, u_B, u_fusion = DS_Combin(alpha, self.opts.num_classes)
        self.evidence_last = (self.alpha_last - 1).reshape(-1, self.opts.resize_size, self.opts.resize_size, self.opts.num_classes).permute(0, 3, 1, 2)

        uncertainty_last = u_last.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)
        uncertainty_A = u_A.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)
        uncertainty_B = u_B.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)
        uncertainty_fusion = u_fusion.reshape(-1, self.opts.resize_size, self.opts.resize_size, 1).permute(0, 3, 1, 2)

        self.pred_A = ((self.evidence_A + 1) / torch.sum(self.evidence_A + 1, dim=1, keepdim=True))
        self.pred_B = ((self.evidence_B + 1) / torch.sum(self.evidence_B + 1, dim=1, keepdim=True))
        self.pred_fusion = ((self.evidence_fusion + 1) / torch.sum(self.evidence_fusion + 1, dim=1, keepdim=True))
        self.pred_last = ((self.evidence_last + 1) / torch.sum(self.evidence_last + 1, dim=1, keepdim=True))

    def update_seg(self, images_a, images_b, masks, epoch):
        self.images_A = images_a
        self.images_B = images_b
        self.masks = masks
        self.current_epoch = epoch
        self.forword_seg()

        # update seg
        self.seg_opt.zero_grad()
        loss_seg = self.backward_seg()
        self.seg_loss = loss_seg.item()
        self.seg_opt.step()

    def backward_seg(self):
        lambda_seg = self.opts.lambda_seg
        loss_seg_A = torch.mean(dce_evidence_u_loss(self.masks.squeeze(1).to(torch.int64), self.alpha_A, self.opts.num_classes,
                                     self.current_epoch, self.opts.lambda_epochs, (self.opts.n_ep + self.opts.n_ep_decay),
                                     self.evidence_A+1))
        loss_seg_B = torch.mean(dce_evidence_u_loss(self.masks.squeeze(1).to(torch.int64), self.alpha_B, self.opts.num_classes,
                                     self.current_epoch, self.opts.lambda_epochs, (self.opts.n_ep + self.opts.n_ep_decay),
                                     self.evidence_B+1))
        loss_seg_fusion = torch.mean(dce_evidence_u_loss(self.masks.squeeze(1).to(torch.int64), self.alpha_fusion, self.opts.num_classes,
                                     self.current_epoch, self.opts.lambda_epochs, (self.opts.n_ep + self.opts.n_ep_decay),
                                     self.evidence_fusion+1))
        loss_seg_last = torch.mean(dce_evidence_u_loss(self.masks.squeeze(1).to(torch.int64), self.alpha_last, self.opts.num_classes,
                                     self.current_epoch, self.opts.lambda_epochs, (self.opts.n_ep + self.opts.n_ep_decay),
                                     self.evidence_last+1))
        self.loss_seg = (loss_seg_A + loss_seg_B + loss_seg_fusion + loss_seg_last) * lambda_seg
        self.loss_seg.backward()
        return self.loss_seg

    def update_lr(self):
        self.disA_sch.step()
        self.disB_sch.step()
        self.disContent_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()
        self.seg_sch.step()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
            self.disB.load_state_dict(checkpoint['disB'])
            self.disContent.load_state_dict(checkpoint['disContent'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        self.seg.load_state_dict(checkpoint['seg'])
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.disB_opt.load_state_dict(checkpoint['disB_opt'])
            self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
            self.seg_opt.load_state_dict(checkpoint['seg_opt'])
        return checkpoint['ep'], checkpoint['total_it'], checkpoint['best_metric']

    def resume_sch(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.disA_sch.load_state_dict(checkpoint['disA_sch'])
        self.disB_sch.load_state_dict(checkpoint['disB_sch'])
        self.disContent_sch.load_state_dict(checkpoint['disContent_sch'])
        self.enc_c_sch.load_state_dict(checkpoint['enc_c_sch'])
        self.enc_a_sch.load_state_dict(checkpoint['enc_a_sch'])
        self.gen_sch.load_state_dict(checkpoint['gen_sch'])
        self.seg_sch.load_state_dict(checkpoint['seg_sch'])

    def save(self, filename, ep, total_it, best_metric):
        state = {
            'disA': self.disA.state_dict(),
            'disB': self.disB.state_dict(),
            'disContent': self.disContent.state_dict(),
            'enc_c': self.enc_c.state_dict(),
            'enc_a': self.enc_a.state_dict(),
            'gen': self.gen.state_dict(),
            'seg': self.seg.state_dict(),
            'disA_opt': self.disA_opt.state_dict(),
            'disB_opt': self.disB_opt.state_dict(),
            'disContent_opt': self.disContent_opt.state_dict(),
            'enc_c_opt': self.enc_c_opt.state_dict(),
            'enc_a_opt': self.enc_a_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'seg_opt': self.seg_opt.state_dict(),
            'disA_sch': self.disA_sch.state_dict(),
            'disB_sch': self.disB_sch.state_dict(),
            'disContent_sch': self.disContent_sch.state_dict(),
            'enc_c_sch': self.enc_c_sch.state_dict(),
            'enc_a_sch': self.enc_a_sch.state_dict(),
            'gen_sch': self.gen_sch.state_dict(),
            'seg_sch': self.seg_sch.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'best_metric': best_metric
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_a = self.normalize_image(self.real_A_encoded).detach()
        images_b = self.normalize_image(self.real_B_encoded).detach()
        images_b2a = self.normalize_image(self.fake_A_encoded).detach()
        images_a2a = self.normalize_image(self.fake_AA_encoded).detach()
        images_a2b = self.normalize_image(self.fake_B_encoded).detach()
        images_b2b = self.normalize_image(self.fake_BB_encoded).detach()
        mask = self.masks.detach()
        predict = self.pred_last.detach()
        predict_f = self.pred_fusion.detach()
        return {"realA": images_a, "fakeA2B": images_a2b, "fakeA2A": images_a2a,
                "realB": images_b, "fakeB2A": images_b2a, "fakeB2B": images_b2b,
                "mask": mask, "predict": predict, "predict_f": predict_f}

    def normalize_image(self, x):
        return x[:, 0:1, :, :]

    def all_loss(self):
        Gen_losses = self.G_loss
        Dis_losses = self.D_loss
        Seg_losses = self.seg_loss
        return Gen_losses, Dis_losses, Seg_losses
