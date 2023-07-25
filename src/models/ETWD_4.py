from .unets_parts import *
from .transformer_partsR import TransformerDown, TransformerDown_HP, TransformerDown_SPrune
from einops import rearrange, repeat
import logging
import gc


def norm(weight, h, w, c):
    weight = rearrange(weight, 'b 1 h w -> b (h w)', h=h, w=w)
    # weight = torch.softmax(weight, dim=1)
    weight = torch.sigmoid(weight)
    weight = rearrange(weight, 'b (h w) -> b 1 h w', h=h, w=w)
    weight = repeat(weight, 'b 1 h w -> b c h w', c=c)
    return weight


class ETWD_decoder(nn.Module):
    def __init__(self, n_classes, scale, bilinear, res):
        super(ETWD_decoder, self).__init__()

        self.n_classes = n_classes
        self.scale = scale
        self.bilinear = bilinear
        self.res = res
        factor = 2 if bilinear else 1

        self.up2 = Up(1024 // self.scale, 512 // factor // self.scale, bilinear)
        self.up3 = Up(512 // self.scale, 256 // factor // self.scale, bilinear)
        self.up4 = Up(256 // self.scale, 128 // factor // self.scale, bilinear)
        self.up5 = Up(128 // self.scale, 64 // self.scale, bilinear)
        self.outc = OutConv(64 // self.scale, self.n_classes)

    def forward(self, x, weight):
        weight0 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)(weight)  # torch.Size([8, 1, 256, 256])  x[0]: torch.Size([8, 16, 256, 256])
        weight1 = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=True)(weight)  # torch.Size([8, 1, 128, 128])  x[1]: torch.Size([8, 32, 128, 128])
        weight2 = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=True)(weight)  # torch.Size([8, 1, 64, 64])    x[2]: torch.Size([8, 64, 64, 64])
        weight3 = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)(weight)  # torch.Size([8, 1, 32, 32])    x[2]: torch.Size([8, 128, 32, 32])

        weight0 = norm(weight0, h=256, w=256, c=16)
        weight1 = norm(weight1, h=128, w=128, c=32)
        weight2 = norm(weight2, h=64 , w=64 , c=64)
        weight3 = norm(weight3, h=32,  w=32,  c=128)

        # print('upsample weight shape: ', weight0.shape, weight1.shape, weight2.shape, x[0].shape, x[1].shape, x[2].shape)
        # # torch.Size([8, 16, 256, 256]) torch.Size([8, 32, 128, 128]) torch.Size([8, 64, 64, 64]) torch.Size([8, 16, 256, 256]) torch.Size([8, 32, 128, 128]) torch.Size([8, 64, 64, 64])

        if self.res == True:
            x0 = weight0 * x[0] + x[0]
            x1 = weight1 * x[1] + x[1]
            x2 = weight2 * x[2] + x[2]
            x3 = weight3 * x[3] + x[3]
        else:
            x0 = weight0 * x[0]
            x1 = weight1 * x[1]
            x2 = weight2 * x[2]
            x3 = weight3 * x[3]

        out = self.up2(x[4], x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.up5(out, x0)
        main_logits = self.outc(out)

        return main_logits


class ETWD_Model(nn.Module):
    def __init__(self, down_block, name, in_channels, n_classes, imgsize, bilinear, res):
        super(ETWD_Model, self).__init__()
        self.name = name
        assert self.name == 'ETWD', 'model load error'
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res = res
        self.scale = 4  # 1 2 4

        self.inc = DoubleConv(in_channels, 64 // self.scale)
        self.down1 = Down(64  // self.scale, 128 // self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, 512 // self.scale)
        self.down4 = Down(512 // self.scale, 1024 // self.scale)

        factor = 2 if bilinear else 1

        self.trans4 = down_block(1024//self.scale, 1024//self.scale, imgsize//16, 4, heads=6, dim_head=256, patch_size=1)  # 256, 1024
        self.conv4 = nn.Conv2d(1024//self.scale, 1024//self.scale//factor, kernel_size=1, padding=0, bias=False)

        self.up1 = ETWD_decoder(self.n_classes, self.scale, self.bilinear, self.res)
        self.up2 = ETWD_decoder(self.n_classes, self.scale, self.bilinear, self.res)
        self.up3 = ETWD_decoder(self.n_classes, self.scale, self.bilinear, self.res)

        message = f"Load ETWD model, self.bilinear is {self.bilinear}"
        print(message); logging.info(message)
        del message; gc.collect()


    def forward(self, x, tar_layer=None, tmp_score=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if tar_layer is not None:
            x5, qkvs1, attns1, fea, tar_attn = self.trans4(x5, tar_layer=tar_layer, tmp_score=tmp_score)  # attn_tensor1: [4, 24, 1024, 1024]
        else:
            x5, qkvs1, attns1, fea = self.trans4(x5, tar_layer=tar_layer, tmp_score=tmp_score)  # attn_tensor1: [4, 24, 1024, 1024]

        x5 = self.conv4(x5)

        attns_tensor = torch.stack(attns1)  #  torch.Size([4, 8, 6, 1024, 1024])
        attns_mean = torch.mean(torch.mean(attns_tensor, dim=0), dim=1)  # torch.Size([8, 4, 6, 1024, 1024]) -> torch.Size([8, 6, 1024, 1024]) -> torch.Size([8, 1024, 1024])
        # print('attns: ', attns_tensor.shape, attns_mean.shape, torch.mean(attns_tensor, dim=0).shape, torch.mean(torch.mean(attns_tensor, dim=0), dim=1).shape)
        # weight = torch.sum(attns_mean, dim=1)  # torch.Size([8, 1024])
        weight = torch.sum(attns_mean, dim=1) / 1024.0 #  torch.Size([8, 1024])
        weight = rearrange(weight, 'b (h w) -> b 1 h w', h=16, w=16)  # torch.Size([8, 1, 32, 32])
        # print('weight[0]: ', weight[0][0])

        out1 = self.up1([x1, x2, x3, x4, x5], weight)
        out2 = self.up2([x1, x2, x3, x4, x5], weight)
        out3 = self.up3([x1, x2, x3, x4, x5], weight)

        if tar_layer is not None:
            return out1, out2, out3, qkvs1, attns1, fea, tar_attn
        else:
            return out1, out2, out3, qkvs1, attns1, fea

def ETWD(**kwargs):
    model = ETWD_Model(TransformerDown_HP, **kwargs)
    return model