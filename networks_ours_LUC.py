import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


####################################################################
# ------------------------- Discriminators --------------------------
####################################################################
class Dis_content(nn.Module):
    def __init__(self):
        super(Dis_content, self).__init__()
        model = []
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs


class MultiScaleDis(nn.Module):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
        super(MultiScaleDis, self).__init__()
        ch = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim * 2, ch, 4, 2, 1, norm, sn)]
        tch = ch
        for _ in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch *= 2
        if sn:
            model += [spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, input, condition):
        outs = []
        x = torch.cat((input, condition), dim=1)
        for Dis in self.Diss:
            outs.append(Dis(x))
            x = self.downsample(x)
        return outs


class Dis(nn.Module):
    def __init__(self, input_dim, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        n_layer = 6
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]  # 16
        tch = ch
        for i in range(1, n_layer - 1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]  # 8
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]  # 2
        tch *= 2
        if sn:
            model += [spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
        return nn.Sequential(*model)

    def cuda(self, gpu):
        self.model.cuda(gpu)

    def forward(self, x_A):
        out_A = self.model(x_A)
        out_A = out_A.view(-1)
        outs_A = []
        outs_A.append(out_A)
        return outs_A


####################################################################
# ---------------------------- Encoders -----------------------------
####################################################################

class E_content(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_dim_a, input_dim_b):
        super(E_content, self).__init__()
        tch = 64
        encA_c = []
        share = []
        self.down_A1 = LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3, norm='Instance')
        self.down_A2 = ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)
        self.down_A3 = ReLUINSConv2d(tch * 2, tch * 4, kernel_size=3, stride=2, padding=1)
        for i in range(0, 3):
            encA_c += [INSResBlock(tch * 4, tch * 4)]
        self.down_A4 = nn.Sequential(*encA_c)

        encB_c = []
        self.down_B1 = LeakyReLUConv2d(input_dim_b, tch, kernel_size=7, stride=1, padding=3, norm='Instance')
        self.down_B2 = ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)
        self.down_B3 = ReLUINSConv2d(tch * 2, tch * 4, kernel_size=3, stride=2, padding=1)
        for i in range(0, 3):
            encB_c += [INSResBlock(tch * 4, tch * 4)]
        self.down_B4 = nn.Sequential(*encB_c)

        for i in range(0, 3):
            share += [INSResBlock(tch * 4, tch * 4)]
        self.share = nn.Sequential(*share)

    def forward(self, xa, xb):
        da1 = self.down_A1(xa)
        da2 = self.down_A2(da1)
        da3 = self.down_A3(da2)
        da4 = self.down_A4(da3)
        da4_share = self.share(da4)
        encoder_features_a = [da1, da2]

        db1 = self.down_B1(xb)
        db2 = self.down_B2(db1)
        db3 = self.down_B3(db2)
        db4 = self.down_B4(db3)
        db4_share = self.share(db4)
        encoder_features_b = [db1, db2]
        return da4_share, db4_share, encoder_features_a, encoder_features_b


# class E_content(nn.Module):
#   def __init__(self, input_dim_a, input_dim_b):
#     super(E_content, self).__init__()
#     encA_c = []
#     tch = 64
#     encA_c += [LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3, norm='Instance')]
#     for i in range(1, 3):
#       encA_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
#       tch *= 2
#     for i in range(0, 3):
#       encA_c += [INSResBlock(tch, tch)]
#
#     encB_c = []
#     tch = 64
#     encB_c += [LeakyReLUConv2d(input_dim_b, tch, kernel_size=7, stride=1, padding=3)]
#     for i in range(1, 3):
#       encB_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
#       tch *= 2
#     for i in range(0, 3):
#       encB_c += [INSResBlock(tch, tch)]
#
#     enc_share = []
#     for i in range(0, 1):
#       enc_share += [INSResBlock(tch, tch)]
#       self.conv_share = nn.Sequential(*enc_share)
#
#     self.convA = nn.Sequential(*encA_c)
#     self.convB = nn.Sequential(*encB_c)
#
#   def forward(self, xa, xb):
#     outputA = self.convA(xa)
#     outputB = self.convB(xb)
#     outputA = self.conv_share(outputA)
#     outputB = self.conv_share(outputB)
#     return outputA, outputB
#
#   def forward_a(self, xa):
#     outputA = self.convA(xa)
#     outputA = self.conv_share(outputA)
#     return outputA
#
#   def forward_b(self, xb):
#     outputB = self.convB(xb)
#     outputB = self.conv_share(outputB)
#     return outputB

class E_attr(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, output_nc=8):
        super(E_attr, self).__init__()
        dim = 64
        self.model_a = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_a, dim, 7, 1),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 2, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, output_nc, 1, 1, 0))
        self.model_b = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_b, dim, 7, 1),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 2, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, output_nc, 1, 1, 0))
        return

    def forward(self, xa, xb):
        xa = self.model_a(xa)
        xb = self.model_b(xb)
        output_A = xa.view(xa.size(0), -1)
        output_B = xb.view(xb.size(0), -1)
        return output_A, output_B

    def forward_a(self, xa):
        xa = self.model_a(xa)
        output_A = xa.view(xa.size(0), -1)
        return output_A

    def forward_b(self, xb):
        xb = self.model_b(xb)
        output_B = xb.view(xb.size(0), -1)
        return output_B


class E_attr_concat(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None):
        super(E_attr_concat, self).__init__()

        ndf = 64
        n_blocks = 4
        max_ndf = 4

        conv_layers_A = [nn.ReflectionPad2d(1)]
        conv_layers_A += [nn.Conv2d(input_dim_a, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_A += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_A += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv_A = nn.Sequential(*conv_layers_A)

        conv_layers_B = [nn.ReflectionPad2d(1)]
        conv_layers_B += [nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv_B = nn.Sequential(*conv_layers_B)

    def forward(self, xa, xb):
        x_conv_A = self.conv_A(xa)
        conv_flat_A = x_conv_A.view(xa.size(0), -1)
        output_A = self.fc_A(conv_flat_A)
        outputVar_A = self.fcVar_A(conv_flat_A)
        x_conv_B = self.conv_B(xb)
        conv_flat_B = x_conv_B.view(xb.size(0), -1)
        output_B = self.fc_B(conv_flat_B)
        outputVar_B = self.fcVar_B(conv_flat_B)
        return output_A, outputVar_A, output_B, outputVar_B

    def forward_a(self, xa):
        x_conv_A = self.conv_A(xa)
        conv_flat_A = x_conv_A.view(xa.size(0), -1)
        output_A = self.fc_A(conv_flat_A)
        outputVar_A = self.fcVar_A(conv_flat_A)
        return output_A, outputVar_A

    def forward_b(self, xb):
        x_conv_B = self.conv_B(xb)
        conv_flat_B = x_conv_B.view(xb.size(0), -1)
        output_B = self.fc_B(conv_flat_B)
        outputVar_B = self.fcVar_B(conv_flat_B)
        return output_B, outputVar_B


####################################################################
# --------------------------- Generators ----------------------------
####################################################################
class G(nn.Module):
    def __init__(self, output_dim_a, output_dim_b, nz):
        super(G, self).__init__()
        self.nz = nz
        ini_tch = 256
        tch_add = ini_tch
        tch = ini_tch
        self.tch_add = tch_add
        self.decA1 = MisINSResBlock(tch, tch_add)
        self.decA2 = MisINSResBlock(tch, tch_add)
        self.decA3 = MisINSResBlock(tch, tch_add)
        self.decA4 = MisINSResBlock(tch, tch_add)

        decA5 = []
        decA5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decA5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decA5 += [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]
        decA5 += [nn.Tanh()]
        self.decA5 = nn.Sequential(*decA5)

        tch = ini_tch
        self.decB1 = MisINSResBlock(tch, tch_add)
        self.decB2 = MisINSResBlock(tch, tch_add)
        self.decB3 = MisINSResBlock(tch, tch_add)
        self.decB4 = MisINSResBlock(tch, tch_add)
        decB5 = []
        decB5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decB5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decB5 += [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)]
        decB5 += [nn.Tanh()]
        self.decB5 = nn.Sequential(*decB5)

        self.mlpA = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, tch_add * 4))
        self.mlpB = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, tch_add * 4))

    def forward_a(self, x, z):
        z = self.mlpA(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decA1(x, z1)
        out2 = self.decA2(out1, z2)
        out3 = self.decA3(out2, z3)
        out4 = self.decA4(out3, z4)
        out = self.decA5(out4)
        return out

    def forward_b(self, x, z):
        z = self.mlpB(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decB1(x, z1)
        out2 = self.decB2(out1, z2)
        out3 = self.decB3(out2, z3)
        out4 = self.decB4(out3, z4)
        out = self.decB5(out4)
        return out


class G_concat(nn.Module):
    def __init__(self, output_dim_a, output_dim_b, nz):
        super(G_concat, self).__init__()
        self.nz = nz
        tch = 256
        dec_share = []
        dec_share += [INSResBlock(tch, tch)]
        self.dec_share = nn.Sequential(*dec_share)

        tch = 256 + self.nz
        # decA1 = []
        # for i in range(0, 3):
        #     decA1 += [INSResBlock(tch, tch)]
        decA1_1 = INSResBlock(tch, tch)
        tch = tch + self.nz
        decA1_2 = INSResBlock(tch, tch)
        tch = tch + self.nz
        decA1_3 = INSResBlock(tch, tch)
        tch = tch + self.nz
        decA2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=4, stride=2, padding=1, output_padding=0)
        tch = tch // 2
        tch = tch + self.nz
        decA3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=4, stride=2, padding=1, output_padding=0)
        tch = tch // 2
        tch = tch + self.nz
        decA4 = [nn.Conv2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
        self.decA1_1 = nn.Sequential(*[decA1_1])
        self.decA1_2 = nn.Sequential(*[decA1_2])
        self.decA1_3 = nn.Sequential(*[decA1_3])
        self.decA2 = nn.Sequential(*[decA2])
        self.decA3 = nn.Sequential(*[decA3])
        self.decA4 = nn.Sequential(*decA4)

        tch = 256 + self.nz
        # decB1 = []
        # for i in range(0, 3):
        #     decB1 += [INSResBlock(tch, tch)]
        decB1_1 = INSResBlock(tch, tch)
        tch = tch + self.nz
        decB1_2 = INSResBlock(tch, tch)
        tch = tch + self.nz
        decB1_3 = INSResBlock(tch, tch)
        tch = tch + self.nz
        decB2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=4, stride=2, padding=1, output_padding=0)
        tch = tch // 2
        tch = tch + self.nz
        decB3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=4, stride=2, padding=1, output_padding=0)
        tch = tch // 2
        tch = tch + self.nz
        decB4 = [nn.Conv2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
        self.decB1_1 = nn.Sequential(*[decB1_1])
        self.decB1_2 = nn.Sequential(*[decB1_2])
        self.decB1_3 = nn.Sequential(*[decB1_3])
        self.decB2 = nn.Sequential(*[decB2])
        self.decB3 = nn.Sequential(*[decB3])
        self.decB4 = nn.Sequential(*decB4)

    def forward_a(self, x, z):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z_1 = torch.cat([out0, z_img], 1)
        out1_1 = self.decA1_1(x_and_z_1)
        x_and_z_2 = torch.cat([out1_1, z_img], 1)
        out1_2 = self.decA1_2(x_and_z_2)
        x_and_z_3 = torch.cat([out1_2, z_img], 1)
        out1 = self.decA1_3(x_and_z_3)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decA2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decA3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decA4(x_and_z4)
        return out4

    def forward_b(self, x, z):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z_1 = torch.cat([out0, z_img], 1)
        out1_1 = self.decB1_1(x_and_z_1)
        x_and_z_2 = torch.cat([out1_1, z_img], 1)
        out1_2 = self.decB1_2(x_and_z_2)
        x_and_z_3 = torch.cat([out1_2, z_img], 1)
        out1 = self.decB1_3(x_and_z_3)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decB2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decB3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decB4(x_and_z4)
        return out4


####################################################################
# --------------------------- Segment ----------------------------
####################################################################
class Seg(nn.Module):
    def __init__(self, output_dim):
        super(Seg, self).__init__()
        ini_tch = 64
        self.num_classes = output_dim

        self.down_a0 = basic_dowm_block(1, ini_tch, stride=1, head_layer=True)
        self.down_a1 = basic_dowm_block(ini_tch * 2, ini_tch * 2)
        self.down_a2 = basic_dowm_block(ini_tch * 2 * 2, ini_tch * 4)
        self.down_a3 = basic_dowm_block(ini_tch * 4 * 2, ini_tch * 8)
        self.down_a4 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_a5 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_a6 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_a7 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_a8 = basic_dowm_block(ini_tch * 8, ini_tch * 8)

        self.up_a8 = basic_up_block(ini_tch * 8, ini_tch * 8)
        self.u_a7 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_a7 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_a6 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_a6 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_a5 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_a5 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_a4 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_a4 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_a3 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_a3 = LDELayer(ini_tch * 8, ini_tch * 4)
        self.u_a2 = layer_uncertainty(ini_tch * 4, output_dim)
        self.up_a2 = LDELayer(ini_tch * 4, ini_tch * 2)
        self.u_a1 = layer_uncertainty(ini_tch * 2, output_dim)
        self.up_a1 = LDELayer(ini_tch * 2, ini_tch)
        self.u_a0 = layer_uncertainty(ini_tch, output_dim)
        self.out_a = LDELayer(ini_tch, output_dim, stride=1, head_layer=True)

        self.down_b0 = basic_dowm_block(1, ini_tch, stride=1, head_layer=True)
        self.down_b1 = basic_dowm_block(ini_tch * 2, ini_tch * 2)
        self.down_b2 = basic_dowm_block(ini_tch * 2 * 2, ini_tch * 4)
        self.down_b3 = basic_dowm_block(ini_tch * 4 * 2, ini_tch * 8)
        self.down_b4 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_b5 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_b6 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_b7 = basic_dowm_block(ini_tch * 8, ini_tch * 8)
        self.down_b8 = basic_dowm_block(ini_tch * 8, ini_tch * 8)

        self.up_b8 = basic_up_block(ini_tch * 8, ini_tch * 8)
        self.u_b7 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_b7 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_b6 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_b6 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_b5 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_b5 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_b4 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_b4 = LDELayer(ini_tch * 8, ini_tch * 8)
        self.u_b3 = layer_uncertainty(ini_tch * 8, output_dim)
        self.up_b3 = LDELayer(ini_tch * 8, ini_tch * 4)
        self.u_b2 = layer_uncertainty(ini_tch * 4, output_dim)
        self.up_b2 = LDELayer(ini_tch * 4, ini_tch * 2)
        self.u_b1 = layer_uncertainty(ini_tch * 2, output_dim)
        self.up_b1 = LDELayer(ini_tch * 2, ini_tch)
        self.u_b0 = layer_uncertainty(ini_tch, output_dim)
        self.out_b = LDELayer(ini_tch, output_dim, stride=1, head_layer=True)

        self.up_8 = basic_up_block(ini_tch * 8 * 2, ini_tch * 8)
        self.UN8 = layer_uncertainty_calibrator(ini_tch * 8)
        self.up_7 = LDELayer_fusion(ini_tch * 8, ini_tch * 8)
        self.UN7 = layer_uncertainty_calibrator(ini_tch * 8)
        self.up_6 = LDELayer_fusion(ini_tch * 8, ini_tch * 8)
        self.UN6 = layer_uncertainty_calibrator(ini_tch * 8)
        self.up_5 = LDELayer_fusion(ini_tch * 8, ini_tch * 8)
        self.UN5 = layer_uncertainty_calibrator(ini_tch * 8)
        self.up_4 = LDELayer_fusion(ini_tch * 8, ini_tch * 8)
        self.UN4 = layer_uncertainty_calibrator(ini_tch * 8)
        self.up_3 = LDELayer_fusion(ini_tch * 8, ini_tch * 4)
        self.UN3 = layer_uncertainty_calibrator(ini_tch * 4)
        self.up_2 = LDELayer_fusion(ini_tch * 4, ini_tch * 2)
        self.UN2 = layer_uncertainty_calibrator(ini_tch * 2)
        self.up_1 = LDELayer_fusion(ini_tch * 2, ini_tch)
        self.UN1 = layer_uncertainty_calibrator(ini_tch)
        self.out = LDELayer_fusion(ini_tch, output_dim, stride=1, head_layer=True)

    def forward(self, content_a, content_b, content_features_a, content_features_b, a, b):
        da0 = self.down_a0(a) # 64,256,256
        da1 = self.down_a1(torch.cat((da0, content_features_a[0]), dim=1)) # 128,128,128
        da2 = self.down_a2(torch.cat((da1, content_features_a[1]), dim=1)) # 256,64,64
        da3 = self.down_a3(torch.cat((da2, content_a), dim=1))  # 512,32,32
        da4 = self.down_a4(da3)  # 512,16,16
        da5 = self.down_a5(da4)  # 512,8,8
        da6 = self.down_a6(da5)  # 512,4,4
        da7 = self.down_a7(da6)  # 512,2,2
        da8 = self.down_a8(da7)  # 512,1,1

        ua7 = self.up_a8(da8)  # 512,2,2
        # uncertainty_a7 = self.u_a7(ua7)
        ua6 = self.up_a7(ua7, da7)  # 512,4,4
        # uncertainty_a6 = self.u_a6(ua6)
        ua5 = self.up_a6(ua6, da6)  # 512,8,8
        # uncertainty_a5 = self.u_a5(ua5)
        ua4 = self.up_a5(ua5, da5)  # 512,16,16
        # uncertainty_a4 = self.u_a4(ua4)
        ua3 = self.up_a4(ua4, da4)  # 512,32,32
        # uncertainty_a3 = self.u_a3(ua3)
        ua2 = self.up_a3(ua3, da3)  # 256,64,64
        uncertainty_a2 = self.u_a2(ua2)
        ua1 = self.up_a2(ua2, da2)  # 128,128,128
        uncertainty_a1 = self.u_a1(ua1)
        ua0 = self.up_a1(ua1, da1)  # 64,256,256
        uncertainty_a0 = self.u_a0(ua0)
        out_a = self.out_a(ua0, da0)  # 2,256,256

        # evidence_a = F.softplus(out_a)
        # alpha_a = evidence_a + 1
        # S_a = torch.sum(alpha_a, dim=1, keepdim=True)
        # uncertainty_a = self.num_classes / S_a

        db0 = self.down_b0(b) # 64,256,256
        db1 = self.down_b1(torch.cat((db0, content_features_b[0]), dim=1)) # 128,128,128
        db2 = self.down_b2(torch.cat((db1, content_features_b[1]), dim=1)) # 256,64,64
        db3 = self.down_b3(torch.cat((db2, content_b), dim=1))  # 512,32,32
        db4 = self.down_b4(db3)  # 512,16,16
        db5 = self.down_b5(db4)  # 512,8,8
        db6 = self.down_b6(db5)  # 512,4,4
        db7 = self.down_b7(db6)  # 512,2,2
        db8 = self.down_b8(db7)  # 512,1,1

        ub7 = self.up_b8(db8)  # 512,2,2
        # uncertainty_b7 = self.u_b7(ub7)
        ub6 = self.up_b7(ub7, db7)  # 512,4,4
        # uncertainty_b6 = self.u_b6(ub6)
        ub5 = self.up_b6(ub6, db6)  # 512,8,8
        # uncertainty_b5 = self.u_b5(ub5)
        ub4 = self.up_b5(ub5, db5)  # 512,16,16
        # uncertainty_b4 = self.u_b4(ub4)
        ub3 = self.up_b4(ub4, db4)  # 512,32,32
        # uncertainty_b3 = self.u_b3(ub3)
        ub2 = self.up_b3(ub3, db3)  # 256,64,64
        uncertainty_b2 = self.u_b2(ub2)
        ub1 = self.up_b2(ub2, db2)  # 128,128,128
        uncertainty_b1 = self.u_b1(ub1)
        ub0 = self.up_b1(ub1, db1)  # 64,256,256
        uncertainty_b0 = self.u_b0(ub0)
        out_b = self.out_b(ub0, db0) # 2,256,256

        # evidence_b = F.softplus(out_b)
        # alpha_b = evidence_b + 1
        # S_b = torch.sum(alpha_b, dim=1, keepdim=True)
        # uncertainty_b = self.num_classes / S_b

        u7 = self.up_8(torch.cat((da8, db8), dim=1))  # 512,2,2
        # u6 = self.up_7(da7, db7, self.UN8(u7, ua7, ub7, uncertainty_a7, uncertainty_b7))  # 512,4,4
        # u5 = self.up_6(da6, db6, self.UN7(u6, ua6, ub6, uncertainty_a6, uncertainty_b6))  # 512,8,8
        # u4 = self.up_5(da5, db5, self.UN6(u5, ua5, ub5, uncertainty_a5, uncertainty_b5))  # 512,16,16
        # u3 = self.up_4(da4, db4, self.UN5(u4, ua4, ub4, uncertainty_a4, uncertainty_b4))  # 512,32,32
        # u2 = self.up_3(da3, db3, self.UN4(u3, ua3, ub3, uncertainty_a3, uncertainty_b3))  # 256,64,64
        u6 = self.up_7(da7, db7, u7)  # 512,4,4
        u5 = self.up_6(da6, db6, u6)  # 512,8,8
        u4 = self.up_5(da5, db5, u5)  # 512,16,16
        u3 = self.up_4(da4, db4, u4)  # 512,32,32
        u2 = self.up_3(da3, db3, u3)  # 256,64,64
        u1 = self.up_2(da2, db2, self.UN3(u2, ua2, ub2, uncertainty_a2, uncertainty_b2))  # 128,128,128
        u0 = self.up_1(da1, db1, self.UN2(u1, ua1, ub1, uncertainty_a1, uncertainty_b1))  # 64,256,256
        out_fusion = self.out(da0, db0, self.UN1(u0, ua0, ub0, uncertainty_a0, uncertainty_b0))  # 2,256,256
        return out_fusion, out_a, out_b, uncertainty_a2, uncertainty_a1, uncertainty_a0, uncertainty_b2, uncertainty_b1, uncertainty_b0



####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += conv3x3(inplanes, outplanes)
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=False)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def conv3x3(in_planes, out_planes):
    return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################

class basic_dowm_block(nn.Module):
    def __init__(self, in_nc, out_nc, stride=2, head_layer=False):
        super(basic_dowm_block, self).__init__()
        if head_layer:
            self.model = nn.Sequential(nn.Conv2d(in_nc, out_nc, 7, stride, 3),
                                       nn.BatchNorm2d(out_nc),
                                       nn.LeakyReLU(0.2, inplace=True))
        else:
            self.model = nn.Sequential(nn.Conv2d(in_nc, out_nc, 3, stride, 1),
                                       nn.BatchNorm2d(out_nc),
                                       nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        return self.model(x)

class basic_up_block(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(basic_up_block, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(in_nc, out_nc, 4, 2, 1),
                                 nn.BatchNorm2d(out_nc),
                                 nn.ReLU(inplace=True))
    def forward(self, x):
        return self.model(x)

class basic_concat_block(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(basic_concat_block, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_nc, out_nc, 1, 1, 0),
                                   nn.BatchNorm2d(out_nc),
                                   nn.ReLU(inplace=True),)
    def forward(self, x):
        return self.model(x)

class Res2Net_dowm_block(nn.Module):
    def __init__(self, in_nc, out_nc, stride=2, head_layer=False):
        super(Res2Net_dowm_block, self).__init__()
        if head_layer:
            self.model = nn.Sequential(nn.Conv2d(in_nc, out_nc, 7, stride, 3),
                                       nn.BatchNorm2d(out_nc),
                                       nn.LeakyReLU(0.2, inplace=True))
        else:
            downsample = nn.Sequential(nn.Conv2d(in_nc, out_nc, 3, stride, 1),
                                   nn.BatchNorm2d(out_nc))
            self.model = Res2NetBottleneck(in_nc, out_nc, downsample=downsample, stride=stride)
    def forward(self, x):
        return self.model(x)

class Res2NetBottleneck(nn.Module):
    expansion = 1  # 残差块的输出通道数=输入通道数*expansion

    def __init__(self, inplanes, planes, downsample=None, upsample=None, stride=1, scales=4, groups=1, norm_layer=True):
        # scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        super(Res2NetBottleneck, self).__init__()

        if norm_layer:  # BN层
            norm_layer = nn.BatchNorm2d

        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        self.upsample = upsample
        # 1*1的卷积层,在第二个layer时缩小/放大图片尺寸
        if downsample:
            bottleneck_planes = groups * planes
            self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=stride)
            self.bn1 = norm_layer(bottleneck_planes)
        else:
            bottleneck_planes = inplanes
            self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1, stride=1)
            self.bn1 = norm_layer(bottleneck_planes)

        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv2 = nn.ModuleList([nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales,
                                              kernel_size=3, stride=1, padding=1, groups=groups) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales - 1)])
        # 1*1的卷积层，经过这个卷积层之后输出的通道数变成
        if upsample:
            self.conv3 = nn.Sequential(nn.Upsample(scale_factor=stride),
                                       nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1))
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.conv3 = nn.Conv2d(bottleneck_planes, planes * self.expansion, kernel_size=1, stride=1)
            self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x

        # 1*1的卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        # scales个(3x3)的残差分层架构
        # xs = []
        # for i in range(self.scales):
        #     channel_resample = out[:, i::4, :, :]  # 每隔4个通道采样一次
        #     xs.append(channel_resample)
        xs = torch.chunk(out, self.scales, 1)  # 将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.leakyrelu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.leakyrelu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        # 1*1的卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        # 下/上采样
        if self.downsample:
            identity = self.downsample(identity)
        elif self.upsample:
            identity = self.upsample(identity)

        out += identity

        out = self.leakyrelu(out)
        return out

class layer_uncertainty(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(layer_uncertainty, self).__init__()
        self.num_classes = outchannel
        self.conv = nn.Conv2d(inchannel, outchannel, 3, 1, 1)

    def forward(self, x):
        out = self.conv(x)
        evidence = F.softplus(out)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S
        return uncertainty

class layer_uncertainty_calibrator(nn.Module):
    def __init__(self, nc):
        super(layer_uncertainty_calibrator, self).__init__()
        self.conva = nn.Conv2d(nc, nc, 1, 1, 0)
        self.convb = nn.Conv2d(nc, nc, 1, 1, 0)
        self.convf = nn.Sequential(nn.Conv2d(nc * 2, nc, 1, 1, 0),
                                   nn.BatchNorm2d(nc),
                                   nn.ReLU(True))
        self.conv_cat = nn.Sequential(nn.Conv2d(nc * 2, nc, 1, 1, 0),
                                     nn.BatchNorm2d(nc),
                                     nn.ReLU(True))
    def forward(self, xf,xa,xb, ua, ub):
        bs,_,w,h = ua.size()
        min_val_ua = ua.view(bs, -1).min(dim=1, keepdim=True)[0].view(bs, 1, 1, 1)
        max_val_ua = ua.view(bs, -1).max(dim=1, keepdim=True)[0].view(bs, 1, 1, 1)
        min_val_ub = ub.view(bs, -1).min(dim=1, keepdim=True)[0].view(bs, 1, 1, 1)
        max_val_ub = ub.view(bs, -1).max(dim=1, keepdim=True)[0].view(bs, 1, 1, 1)
        ua = (ua - min_val_ua) / (max_val_ua - min_val_ua) * (1 - 1e-4) + 1e-4
        ub = (ub - min_val_ub) / (max_val_ub - min_val_ub) * (1 - 1e-4) + 1e-4
        a = xa + self.conva(xa) * ua.detach()
        b = xb + self.convb(xb) * ub.detach()
        f = self.convf(torch.cat((a, b), dim=1))
        out = self.conv_cat(torch.cat((xf, f), dim=1))
        return out


class uncertainty_navigator(nn.Module):
    def __init__(self, nc, n_layers):
        super(uncertainty_navigator, self).__init__()
        self.n_layers = n_layers
        scale_factor = 1 / (2**(n_layers-1))

        self.downsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conva = nn.Conv2d(nc, nc, 1, 1, 0)
        self.convb = nn.Conv2d(nc, nc, 1, 1, 0)
        self.convf = nn.Sequential(nn.Conv2d(nc * 2, nc, 1, 1, 0),
                                   nn.BatchNorm2d(nc),
                                   nn.ReLU(True))
        self.conv_cat = nn.Sequential(nn.Conv2d(nc * 2, nc, 1, 1, 0),
                                     nn.BatchNorm2d(nc),
                                     nn.ReLU(True))
    def forward(self, xf,xa,xb, ua, ub):
        a = xa + self.conva(xa) * self.downsample(ua.detach())
        b = xb + self.conva(xb) * self.downsample(ub.detach())
        f = self.convf(torch.cat((a, b), dim=1))
        out = self.conv_cat(torch.cat((xf, f), dim=1))
        return out

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape),
                                self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [
                spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0)]
        model += [nn.InstanceNorm2d(n_out)]
        model += [nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU()]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class MisINSResBlock(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out


class GaussianNoiseLayer(nn.Module):
    def __init__(self, ):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise


class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding, bias=True)]
        # model += [nn.Upsample(scale_factor=2)]
        # model += [nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1)]
        model += [nn.InstanceNorm2d(n_out)]
        model += [nn.ReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class ReLUBNConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUBNConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding, bias=True)]
        model += [nn.BatchNorm2d(n_out)]
        model += [nn.ReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class SCELayer(nn.Module):
    def __init__(self, inchannel, ratio=16, spatial_kernel=7):
        super(SCELayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(inchannel, inchannel // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel // ratio, inchannel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
                                  nn.Sigmoid())

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.conv(torch.cat([max_out, avg_out], dim=1))
        spatial_out = spatial_att * x
        max_out = self.mlp(self.max_pool(spatial_out))
        avg_out = self.mlp(self.avg_pool(spatial_out))
        channel_att = self.sigmoid(max_out + avg_out)
        out = channel_att * spatial_out
        return out


class LDELayer(nn.Module):
    def __init__(self, inchannel, outchannel, ratio=16, spatial_kernel=7, stride=2, head_layer=False):
        super(LDELayer, self).__init__()
        self.SCE_1 = SCELayer(inchannel, ratio, spatial_kernel)
        self.Conv = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, 1, 0),
                                  nn.BatchNorm2d(inchannel),
                                  nn.ReLU(inplace=True))
        self.SCE_2 = SCELayer(inchannel, ratio, spatial_kernel)
        self.SCE_3 = SCELayer(inchannel, ratio, spatial_kernel)
        if head_layer:
            self.UpConv = nn.Sequential(nn.Conv2d(inchannel * 2, outchannel, 3, 1, 1))
        else:
            # self.UpConv = Res2Net_up_block(inchannel * 2, outchannel, stride)
            self.UpConv = basic_up_block(inchannel * 2, outchannel)

    def forward(self, x, y):
        y = self.Conv(self.SCE_1(y))
        out = self.UpConv(torch.cat((self.SCE_2(x), self.SCE_3(y)), dim=1))
        return out

class LDELayer_fusion(nn.Module):
    def __init__(self, inchannel, outchannel, ratio=16, spatial_kernel=7, stride=2, head_layer=False):
        super(LDELayer_fusion, self).__init__()
        self.SCE_1 = SCELayer(inchannel, ratio, spatial_kernel)
        self.SCE_2 = SCELayer(inchannel, ratio, spatial_kernel)
        self.Conv = nn.Sequential(nn.Conv2d(inchannel * 2, inchannel, 1, 1, 0),
                                  nn.BatchNorm2d(inchannel),
                                  nn.ReLU(inplace=True))
        self.SCE_3 = SCELayer(inchannel, ratio, spatial_kernel)
        self.SCE_4 = SCELayer(inchannel, ratio, spatial_kernel)
        if head_layer:
            self.UpConv = nn.Sequential(nn.Conv2d(inchannel * 2, outchannel, 3, 1, 1))
        else:
            # self.UpConv = Res2Net_up_block(inchannel * 2, outchannel, stride)
            self.UpConv = basic_up_block(inchannel * 2, outchannel)

    def forward(self, x1, x2, y):
        x = self.Conv(torch.cat((self.SCE_1(x1), self.SCE_2(x2)), dim=1))
        out = self.UpConv(torch.cat((self.SCE_3(x), self.SCE_4(y)), dim=1))
        return out


####################################################################
# --------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        # BCE Loss for each class
        bce_loss = 0
        targets = targets.squeeze(1)
        for class_idx in range(inputs.size(1)):
            target_class = (targets == class_idx).float()
            bce_loss += F.binary_cross_entropy(inputs[:, class_idx, :, :], target_class, reduction='mean')

        # Dice Coefficient for each class
        dice_loss = 0
        for class_idx in range(inputs.size(1)):
            input_class = inputs[:, class_idx, :, :]
            target_class = (targets == class_idx).float()

            intersection = (input_class * target_class).sum()
            dice_loss += 1 - ((2. * intersection + smooth) / (input_class.sum() + target_class.sum() + smooth))

        # Combine the two losses
        loss = bce_loss + dice_loss

        return loss

class cosine_similarity_loss(nn.Module):
    def __init__(self):
        super(cosine_similarity_loss, self).__init__()
    def forward(self, matrix1, matrix2):
        matrix1_flat = matrix1.view(matrix1.size(0), -1)
        matrix2_flat = matrix2.view(matrix2.size(0), -1)
        cos_sim = F.cosine_similarity(matrix1_flat, matrix2_flat, dim=1)
        sim_loss = -cos_sim
        return sim_loss

def DS_Combin(alpha, classes):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """
    def DS_Combin_two(alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = classes/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a, u_a, u[0], u[1]

    for v in range(len(alpha)-1):
        if v==0:
            alpha_a, _, uct, upet = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a, u_a, _, ufusion = DS_Combin_two(alpha_a, alpha[v+1])
    return alpha_a, u_a, uct, upet, ufusion

