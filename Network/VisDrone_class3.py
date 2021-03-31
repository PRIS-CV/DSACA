import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import math
from torchvision import models
# __all__ = ['vgg19']
# model_urls = {
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
# }

def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
         int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
    return d1


class VGG(nn.Module):
    def __init__(self, load_weights=False):
        super(VGG, self).__init__()

        self.seen = 0
        self.cfg = {
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],
        'Spatial': [(512,256,1,0,1,0),(256,128,3,1,1,1),
                    (512,256,1,0,1,0),(256,128,3,2,1,2),
                    (512,256,1,0,1,0),(256,128,3,3,1,3),
                    (512,256,1,0,1,0),(256,128,3,4,1,4)],
        'Fusion' : [(512,128,1,0,1,0),(128,64,3,1,1,0),(64,32,3,1,1,0),(32,6,1,0,1,0)],
        }#可加动态卷积

        self.features = make_layers(self.cfg['E'])
        self.spatial = make_mcc_layers(self.cfg['Spatial'])
        self.fusion = make_mcc_layers(self.cfg['Fusion'])

        self.reg1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.reg2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.reg3 = nn.Sequential(
            nn.Conv2d(128, 3, 1),
        )

        self.dialited_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=5, dilation=5),
            nn.ReLU(inplace=True),
        )
        self.dialited_conv4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
        )
        self.dialited_conv5 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(256*3, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
        )

        self.out2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1),
        )



        self.upscore2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upscore8 = nn.UpsamplingBilinear2d(scale_factor=8)



        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.features.state_dict().items())):
                list(self.features.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][
                                                                             1].data[:]

    def forward(self, x, gt):
        gt = x.clone()

        pd = (8, 8, 8, 8)
        x = F.pad(x, pd, 'constant')


        conv1 = self.features[0:4](x)
        conv2 = self.features[4:9](conv1)
        conv3 = self.features[9:16](conv2)
        conv4 = self.features[16:23](conv3)
        conv5 = self.features[23:28](conv4)

        d1 = self.spatial[0:4](conv5)
        d2 = self.spatial[4:8](conv5)
        d3 = self.spatial[8:12](conv5)
        d4 = self.spatial[12:16](conv5)
        dcon_x = torch.cat((d1, d2, d3, d4), 1)
        mask_pre = self.fusion(dcon_x).type(torch.FloatTensor).cuda()

        mask_pre = mask_pre.type(torch.FloatTensor).cuda()
        mask_pre = self.upscore8(mask_pre)
        mask_pre = crop(mask_pre, gt)

        p3 = conv3
        p4 = conv4
        p5 = conv5

        d3 = self.dialited_conv3(p3)
        d4 = self.upscore2(self.dialited_conv4(p4))
        d5 = self.upscore2(self.dialited_conv5(p5))


        ''' attention '''
        #print(p3.shape, p4.shape,d3.shape, d4.shape, d5.shape)
        d3 = crop(d3, d4)
        feature = torch.cat(( d3, d4, d5), 1)
        feature = self.fuse(feature)

        out2 = self.upscore4(self.out2(feature))
        #print(out2.shape)
        out2 = crop(out2, gt)

        out = self.upscore2(conv5)
        out = self.reg1(out)
        out = out + feature
        out = self.reg2(out)
        out = self.reg3(out)
        out = self.upscore4(out)

        out1 = crop(out, gt)
        return out1, out2, mask_pre


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_mcc_layers(cfg, batch_norm=False):
    layers = []
    for v in cfg:
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # input, output, kernel_size, padding, stride, dilation
            if v[5]:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], padding=v[3], stride=v[4],
                                   dilation=v[5])
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], padding=v[3], stride=v[4],)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


# def vgg19():
#     """VGG 19-layer model (configuration "E")
#         model pre-trained on ImageNet
#     """
#     model = VGG(make_layers(cfg['E']))
#     model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
#     return model