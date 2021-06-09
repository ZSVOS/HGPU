
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import libs.utils.ConvGRU2 as ConvGRU

ws = 512


class EncoderNet(nn.Module):
    flag = 'pre'
    def __init__(self):
        super(EncoderNet, self).__init__()
        # image1
        resnet_im1 = models.resnet101(pretrained=True)
        self.conv1_1 = resnet_im1.conv1
        self.bn1_1 = resnet_im1.bn1
        self.relu_1 = resnet_im1.relu
        self.maxpool_1 = resnet_im1.maxpool

        self.res2_1 = resnet_im1.layer1
        self.res3_1 = resnet_im1.layer2
        self.res4_1 = resnet_im1.layer3
        self.res5_1 = resnet_im1.layer4

        # image2
        resnet_im2 = models.resnet101(pretrained=True)
        self.conv1_2 = resnet_im2.conv1
        self.bn1_2 = resnet_im2.bn1
        self.relu_2 = resnet_im2.relu
        self.maxpool_2 = resnet_im2.maxpool

        self.res2_2 = resnet_im2.layer1
        self.res3_2 = resnet_im2.layer2
        self.res4_2 = resnet_im2.layer3
        self.res5_2 = resnet_im2.layer4

        # flow
        resnet_fl = models.resnet101(pretrained=True)
        self.conv1_3 = resnet_fl.conv1
        self.bn1_3 = resnet_fl.bn1
        self.relu_3 = resnet_fl.relu
        self.maxpool_3 = resnet_fl.maxpool

        self.res2_3 = resnet_fl.layer1
        self.res3_3 = resnet_fl.layer2
        self.res4_3 = resnet_fl.layer3
        self.res5_3 = resnet_fl.layer4

        # update
        # self.CG2 = ConvGRU.ConvGRUCell(input_size=256, hidden_size=256)
        self.CG3 = ConvGRU.ConvGRUCell(input_size=512, hidden_size=512)
        self.CG4 = ConvGRU.ConvGRUCell(input_size=1024, hidden_size=1024)
        self.CG5 = ConvGRU.ConvGRUCell(input_size=2048, hidden_size=2048)


        # macu
        self.gac_2 = MACU(128*2)
        self.gac_3 = MACU(256*2)
        self.gac_4 = MACU(512*2)
        self.gac_5 = MACU(1024*2)

        # message
        self.pca_res2 = MessageAgg(channel=256)
        self.pca_res3 = MessageAgg(channel=512)
        self.pca_res4 = MessageAgg(channel=1024)
        self.pca_res5 = MessageAgg(channel=2048)

        # readout
        self.cca_res2 = Readout(channel=256)
        self.cca_res3 = Readout(channel=512)
        self.cca_res4 = Readout(channel=1024)
        self.cca_res5 = Readout(channel=2048)

    # f1: image1 f2: image2 f3: flow
    def forward_res2(self, f1, f2, f3):
        x1 = self.conv1_1(f1)
        x1 = self.bn1_1(x1)
        x1 = self.relu_1(x1)
        x1 = self.maxpool_1(x1)
        h2_1 = self.res2_1(x1)

        x2 = self.conv1_2(f2)
        x2 = self.bn1_2(x2)
        x2 = self.relu_2(x2)
        x2 = self.maxpool_2(x2)
        h2_2 = self.res2_2(x2)

        x3 = self.conv1_3(f3)
        x3 = self.bn1_3(x3)
        x3 = self.relu_3(x3)
        x3 = self.maxpool_3(x3)
        h2_3 = self.res2_3(x3)

        return h2_1, h2_2, h2_3

    # f1: image1 f2: image2 f3: flow
    def forward(self, f1, f2, f3):
        h2_1, h2_2, h2_3 = self.forward_res2(f1, f2, f3)
        Q2_1 = self.cca_res2(h2_1, h2_3)
        Q2_2 = self.cca_res2(h2_2, h2_3)
        Q2_3 = self.cca_res2(h2_1, h2_2)

        # res3 layer: message
        h3_1 = self.res3_1(Q2_1)
        h3_2 = self.res3_2(Q2_2)
        h3_3 = self.res3_3(Q2_3)

        Za1, Zb1 = self.pca_res3(h3_1, h3_3)
        Za2, Zb2 = self.pca_res3(h3_2, h3_3)
        Za3, Zb3 = self.pca_res3(h3_1, h3_2)

        # res3 layer: update
        g1_3 = self.CG3(Za1, h3_1)
        g3_1 = self.CG3(Zb1, h3_3)
        g2_3 = self.CG3(Za2, h3_2)
        g3_2 = self.CG3(Zb2, h3_3)
        g1_2 = self.CG3(Za3, h3_1)
        g2_1 = self.CG3(Zb3, h3_2)

        # res3 layer: readout
        Q3_1 = self.cca_res3(g1_3, g3_1)
        Q3_2 = self.cca_res3(g2_3, g3_2)
        Q3_3 = self.cca_res3(g1_2, g2_1)


        # res4 layer: message
        h4_1 = self.res4_1(Q3_1)
        h4_2 = self.res4_2(Q3_2)
        h4_3 = self.res4_3(Q3_3)

        Za1, Zb1  = self.pca_res4(h4_1, h4_3)
        Za2, Zb2 = self.pca_res4(h4_2, h4_3)
        Za3, Zb3 = self.pca_res4(h4_1, h4_2)

        # res4 layer: update
        g1_3 = self.CG4(Za1, h4_1)
        g3_1 = self.CG4(Zb1, h4_3)
        g2_3 = self.CG4(Za2, h4_2)
        g3_2 = self.CG4(Zb2, h4_3)
        g1_2 = self.CG4(Za3, h4_1)
        g2_1 = self.CG4(Zb3, h4_2)

        # res4 layer: readout
        Q4_1 = self.cca_res4(g1_3, g3_1)
        Q4_2 = self.cca_res4(g2_3, g3_2)
        Q4_3 = self.cca_res4(g1_2, g2_1)

        # res5 layer: message
        h5_1 = self.res5_1(Q4_1)
        h5_2 = self.res5_2(Q4_2)
        h5_3 = self.res5_3(Q4_3)

        Za1, Zb1 = self.pca_res5(h5_1, h5_3)
        Za2, Zb2 = self.pca_res5(h5_2, h5_3)
        Za3, Zb3 = self.pca_res5(h5_1, h5_2)

        # res5 layer: update
        g1_3 = self.CG5(Za1, h5_1)
        g3_1 = self.CG5(Zb1, h5_3)
        g2_3 = self.CG5(Za2, h5_2)
        g3_2 = self.CG5(Zb2, h5_3)
        g1_2 = self.CG5(Za3, h5_1)
        g2_1 = self.CG5(Zb3, h5_2)

        # res5 layer: readout
        Q5_1 = self.cca_res5(g1_3, g3_1)
        Q5_2 = self.cca_res5(g2_3, g3_2)
        Q5_3 = self.cca_res5(g1_2, g2_1)

        # : image1 and flow
        h5_v1 = self.gac_5(Q5_1)
        h4_v1 = self.gac_4(Q4_1)
        h3_v1 = self.gac_3(Q3_1)
        h2_v1 = self.gac_2(Q2_1)

        # : image2 and flow
        h5_v2 = self.gac_5(Q5_2)
        h4_v2 = self.gac_4(Q4_2)
        h3_v2 = self.gac_3(Q3_2)
        h2_v2 = self.gac_2(Q2_2)

        # : image1 and image2
        h5_v3 = self.gac_5(Q5_3)
        h4_v3 = self.gac_4(Q4_3)
        h3_v3 = self.gac_3(Q3_3)
        h2_v3 = self.gac_2(Q2_3)

        return h5_v1, h4_v1, h3_v1, h2_v1, h5_v2, h4_v2, h3_v2, h2_v2, h5_v3, h4_v3, h3_v3, h2_v3


# Message
class MessageAgg(nn.Module):
    def __init__(self, channel):
        super(MessageAgg, self).__init__()
        # project c-dimensional features to multiple lower dimensional spaces
        channel_low = channel // 16

        self.p_f1 = nn.Conv2d(channel, channel_low, kernel_size=1)  # p_f1: project image features
        self.p_f2 = nn.Conv2d(channel, channel_low, kernel_size=1)  # p_f2: project flow features

        self.c_f1 = nn.Conv2d(channel, 1, kernel_size=1)   # c_f1: transform image features to a map by conv 1x1
        self.c_f2 = nn.Conv2d(channel, 1, kernel_size=1)    # c_f2: transform flow features to a map by conv 1x1

        self.relu = nn.ReLU()

    # f1: image, f2: flow
    def forward(self, f1, f2):
        flag = EncoderNet.flag
        if flag == 'pre':
            # Stack 1
            f1_1, f2_1 = self.forward_sa(f1, f2)            # soft attention
            f1_hat, f2_hat = self.forward_ca(f1_1, f2_1)    # co-attention
        else:
            # m 1
            f1_1, f2_1 = self.forward_sa(f1, f2)  # soft attention
            f1_hat, f2_hat = self.forward_ca(f1_1, f2_1)  # co-attention

            fp1_hat = F.relu(f1_hat + f1)
            fp2_hat = F.relu(f2_hat + f2)

            # m 2
            f1_2, f2_2 = self.forward_sa(fp1_hat, fp2_hat)
            f1_hat, f2_hat = self.forward_ca(f1_2, f2_2)

            fp1_hat = F.relu(f1_hat + fp1_hat)
            fp2_hat = F.relu(f2_hat + fp2_hat)

            # m 3
            f1_3, f2_3 = self.forward_sa(fp1_hat, fp2_hat)
            f1_hat, f2_hat = self.forward_ca(f1_3, f2_3)

            fp1_hat = F.relu(f1_hat + fp1_hat)
            fp2_hat = F.relu(f2_hat + fp2_hat)

            # m 4
            f1_4, f2_4 = self.forward_sa(fp1_hat, fp2_hat)
            f1_hat, f2_hat = self.forward_ca(f1_4, f2_4)

            fp1_hat = F.relu(f1_hat + fp1_hat)
            fp2_hat = F.relu(f2_hat + fp2_hat)

            # m 5
            f1_5, f2_5 = self.forward_sa(fp1_hat, fp2_hat)
            f1_hat, f2_hat = self.forward_ca(f1_5, f2_5)

        return f1_hat, f2_hat

    # Soft Attention, f1: image and f2: flow
    def forward_sa(self, f1, f2):

        c1 = self.c_f1(f1)  # channel -> 1
        c2 = self.c_f2(f2)  # channel -> 1

        n, c, h, w = c1.shape
        c1 = c1.view(-1, h*w)
        c2 = c2.view(-1, h*w)

        c1 = F.softmax(c1, dim=1)
        c2 = F.softmax(c2, dim=1)

        c1 = c1.view(n, c, h, w)
        c2 = c2.view(n, c, h, w)

        # '*' indicates Hadamard product
        f1_sa = c1 * f1
        f2_sa = c2 * f2

        # f1_sa and f2_sa indicate attention-enhanced features of image and flow, respectively
        return f1_sa, f2_sa

    # f1: image and f2: flow
    def forward_ca(self, f1, f2):

        f1_cl = self.p_f1(f1)   # f1_cl: dimension from channel to channel_low
        f2_cl = self.p_f2(f2)   # f2_cl: dimension from channel to channel_low

        N, C, H, W = f1_cl.shape
        f1_cl = f1_cl.view(N, C, H * W)
        f2_cl = f2_cl.view(N, C, H * W)
        f2_cl = torch.transpose(f2_cl, 1, 2)

        # Affinity matrix: edge features
        A = torch.bmm(f2_cl, f1_cl)

        # A_r: softmax row, A_c: softmax col
        A_c = torch.tanh(A)
        A_r = torch.transpose(A_c, 1, 2)

        N, C, H, W = f1.shape

        f1_v = f1.view(N, C, H * W)
        f2_v = f2.view(N, C, H * W)

        # e_tu and e_ut
        f1_hat = torch.bmm(f1_v, A_r)
        f2_hat = torch.bmm(f2_v, A_c)
        f1_hat = f1_hat.view(N, C, H, W)
        f2_hat = f2_hat.view(N, C, H, W)

        f1_hat = F.normalize(f1_hat)
        f2_hat = F.normalize(f2_hat)

        return f1_hat, f2_hat


# motion-appearance readout
class Readout(nn.Module):
    def __init__(self, channel):
        super(Readout, self).__init__()

        self.gac = MACU(channel)
        self.gaf = FR(channel)

    def forward(self, f1, f2):
        a1 = self.gac(f1)
        a2 = self.gac(f2)
        aff = self.gaf(a1, a2)
        return aff


# Fusion
class FR(nn.Module):
    def __init__(self, channels=64, r=4):
        super(FR, self).__init__()
        inter_channels = int(channels // r)

        # local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # local attention
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # global attention
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.sigmoid = torch.sigmoid()
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


# motion-appearance context updating
class MACU(nn.Module):
    def __init__(self, input_channels, eps=1e-5):
        super(MACU, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, input_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
        self.epsilon = eps

    def forward(self, x):
        Nl = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon)
        glo = Nl.pow(0.5) * self.alpha
        Nc = (glo.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        cal = self.gamma / Nc

        v_fea = x * 1. + x * torch.tanh(glo * cal + self.beta)
        return v_fea


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MotionRef(nn.Module):
    def __init__(self, in_channel):
        super(MotionRef, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1,
                               padding=1)
        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid = torch.sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        mdim = 256
        self.GC = GC(2048+1, mdim)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(1024+1, mdim)
        self.RF3 = Refine(512+1, mdim)
        self.RF2 = Refine(256+1, mdim)

        self.pred5 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.concat = nn.Conv2d(4, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

        self.ctr5 = MotionRef(2048)
        self.ctr4 = MotionRef(1024)
        self.ctr3 = MotionRef(512)
        self.ctr2 = MotionRef(256)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2, h5_3, h4_3, h3_3, h2_3):
        # mask1
        c5_1 = self.ctr5(h5_3)
        c4_1 = self.ctr4(h4_3)
        c3_1 = self.ctr3(h3_3)
        c2_1 = self.ctr2(h2_3)

        c2_up_1 = F.interpolate(c2_1, size=(ws, ws), mode='bilinear', align_corners=False)
        c3_up_1 = F.interpolate(c3_1, size=(ws, ws), mode='bilinear', align_corners=False)
        c4_up_1 = F.interpolate(c4_1, size=(ws, ws), mode='bilinear', align_corners=False)
        c5_up_1 = F.interpolate(c5_1, size=(ws, ws), mode='bilinear', align_corners=False)

        concat_1 = torch.cat([c2_up_1, c3_up_1, c4_up_1, c5_up_1], dim=1)
        c_1 = self.concat(concat_1)
        c_1 = torch.sigmoid(c_1)

        h5_1 = torch.cat((h5_1, c5_1), dim=1)
        h4_1 = torch.cat((h4_1, c4_1), dim=1)
        h3_1 = torch.cat((h3_1, c3_1), dim=1)
        h2_1 = torch.cat((h2_1, c2_1), dim=1)

        m_1 = self.forward_mask(h5_1, h4_1, h3_1, h2_1)

        # mask2
        h5_2 = torch.cat((h5_2, c5_1), dim=1)
        h4_2 = torch.cat((h4_2, c4_1), dim=1)
        h3_2 = torch.cat((h3_2, c3_1), dim=1)
        h2_2 = torch.cat((h2_2, c2_1), dim=1)

        m_2 = self.forward_mask(h5_2, h4_2, h3_2, h2_2)

        m_1 = m_1 * c_1
        m_2 = m_2 * c_1

        return m_1, m_2

    def forward_mask(self, x, r4, r3, r2):
        x = self.GC(x)
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
        m5 = x + r
        m4 = self.RF4(r4, m5)
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2))
        p2_up = F.interpolate(p2, size=(ws, ws), mode='bilinear', align_corners=False)
        pred = torch.sigmoid(p2_up)

        return pred


class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class AtrousBlock(nn.Module):
    def __init__(self, inplanes, planes, rate, stride=1):
        super(AtrousBlock, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              dilation=rate, padding=rate)

    def forward(self, x):
        return self.conv(x)


class PyramidDilationConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(PyramidDilationConv, self).__init__()

        rate = [6, 12, 18]

        self.block0 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.block1 = AtrousBlock(inplanes, planes, rate[0])
        self.block2 = AtrousBlock(inplanes, planes, rate[1])
        self.block3 = AtrousBlock(inplanes, planes, rate[2])
        self.bn = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block1(x)
        x3 = self.block1(x)

        xx = torch.cat([x0, x1, x2, x3], dim=1)
        xx = self.bn(xx)
        return xx


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

        out_planes = int(planes / 4)
        self.pdc = PyramidDilationConv(inplanes, out_planes)

    def forward(self, f, pm):
        s = self.pdc(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        m = s + F.interpolate(pm, size=s.shape[2:4], mode='bilinear', align_corners=False)

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m
