import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
np.set_printoptions(threshold=1000)
import cv2
import random
#from utils.visualization import featuremap_visual, featuremap1d_visual

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def heat_pos_embed(height=16, width=16, sigma=0.2): #0.6
    heatmap = np.zeros((1, height*width, height, width))
    factor = 1/(2*sigma*sigma)
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x)/width) ** 2
            y_vec = ((np.arange(0, height) - y)/height) ** 2
            xv, yv = np.meshgrid(x_vec, y_vec)
            exponent = factor * (xv + yv)
            exp = np.exp(-exponent)
            heatmap[0, y*height + x, :, :] = exp
    return heatmap

def pos_grid(height=32, width=32):
    grid = np.zeros((1, height * width, height, width))
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x) / width) ** 2
            y_vec = ((np.arange(0, height) - y) / height) ** 2
            yv, xv = np.meshgrid(x_vec, y_vec)
            disyx = (yv + xv)
            grid[0, y * width + x, :, :] = disyx
    return grid

def pos_grid_mask(height=32, width=32, thresh=0.25):
    grid = np.zeros((1, height * width, height, width))
    for y in range(height):
        for x in range(width):
            x_vec = ((np.arange(0, width) - x) / width) ** 2
            y_vec = ((np.arange(0, height) - y) / height) ** 2
            yv, xv = np.meshgrid(x_vec, y_vec)
            disyx = (yv + xv)
            disyx[disyx > thresh] = -1
            disyx[disyx >= 0] = 1
            disyx[disyx == -1] = 0
            grid[0, y * width + x, :, :] = disyx
    return grid

def relative_pos_index(height=32, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += height - 1  # shift to start from 0 # the relative pos in y axial
    relative_coords[:, :, 1] += weight - 1  # the relative pos in x axial
    relative_coords[:, :, 0] *= 2*weight - 1 # the 1d pooling pos to recoard the pos
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index

def relative_pos_index_dis(height=32, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    relative_coords[:, :, 0] += height - 1  # shift to start from 0 # the relative pos in y axial
    relative_coords[:, :, 1] += weight - 1  # the relative pos in x axial
    relative_coords[:, :, 0] *= 2*weight - 1 # the 1d pooling pos to recoard the pos
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index, dis

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, tar_layer=None, tmp_score=None, **kwargs):
        return self.fn(self.norm(x), tar_layer=tar_layer, tmp_score=tmp_score, **kwargs)

class PreNorm2pm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, prob, **kwargs):
        return self.fn(self.norm(x), prob, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, tar_layer=None, tmp_score=None):
        # tar_layer=None and tmp_score are not used in FeedForward, and these two entries are added because in order to be able to add tar_layer and tmp_score to Attention_RPEHP, you have to add them to PreNorm. FeedForward and Attention_RPEHP share the same PreNorm in Transformer_HP,
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), torch.cat((q, k, v), dim=-1), attn #attn

class Attention_RPEHP(nn.Module):
    def __init__(self, dim, heads=6, dim_head=256, dropout=0., height=16, width=16):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) # √

        relative_position_index, dis = relative_pos_index_dis(height, width)
        self.relative_position_index = relative_position_index # hw hw
        self.dis = dis.cuda()  # hw hw
        self.headsita = nn.Parameter(torch.zeros(heads) + torch.range(1, heads)*0.1, requires_grad=True) # √
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), heads), requires_grad=True) # √
        self.height = height
        self.weight = width

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # √

    def forward(self, x, tar_layer=None, tmp_score=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # tuple, 每一个的为torch.Size([8, 256, 1536])
        # print('qkv: ', len(qkv), qkv[0].shape, qkv[1].shape, qkv[2].shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # torch.Size([8, 6, 256, 256])
        # print('q, k, v: ', q.shape, k.shape, v.shape)
        dots0 = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.height * self.weight, self.height * self.weight, -1)  # n n h
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        factor = 1 / (2 * self.headsita**2 + 1e-10)  # g
        exponent = factor[:, None, None] * self.dis[None, :, :] # g hw hw
        pos_embed = torch.exp(-exponent)  # g hw hw
        dots = dots0 + relative_position_bias.unsqueeze(0) + 0.01*pos_embed[None, :, :, :]

        attn = self.attend(dots)

        # print('Attention_RPEHP 1 tar_layer: ', tar_layer)

        if tar_layer is not None:  # 非训练模型过程，正处于计算重要性中
            # print('Attention_RPEHP 2 tar_layer: ', tar_layer)
            tar_attn = attn
            if tmp_score is not None:
                attn = tmp_score
        else:  # 训练模型阶段
            tar_attn = None


        out = torch.matmul(attn, v)  # torch.Size([8, 6, 256, 256])
        # print('out: ', out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')  # torch.Size([8, 256, 1536])
        # print('out: ', out.shape)

        if tar_layer is not None:  # 非训练模型过程，正处于计算重要性中
            # print('Attention_RPEHP 3 tar_layer: ', tar_layer)
            return self.to_out(out), torch.cat((q, k, v), dim=-1), self.attend(dots0), tar_attn

        else:  # 训练模型阶段
            return self.to_out(out), torch.cat((q, k, v), dim=-1), self.attend(dots0)
        #return self.to_out(out), self.attend(dots0)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        qkvs, attns = [], []
        for attn, ff in self.layers:
            ax, qkv, attn = attn(x)
            qkvs.append(qkv)
            attns.append(attn)
            x = ax + x
            x = ff(x) + x
        return x, qkvs, attns


class Transformer_HP(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, image_height, patch_height, mlp_dim=1024, dropout=0., height=16, width=16):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_RPEHP(dim, heads=heads, dim_head=dim_head, dropout=dropout, height=height, width=width)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.recover1_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
        )

    def forward(self, x, tar_layer=None, tmp_score=None):
        qkvs, attns, fea = [], [], []
        count_layer = 0

        if tar_layer is not None and tar_layer < 0:
            tar_layer = abs(tar_layer) - 1

        for attn, ff in self.layers:

            if count_layer == tar_layer:
                # print('Transformer_HP 1 tar_layer: ', tar_layer)
                ax, qkv, attn, tar_attn = attn(x, tar_layer, tmp_score)
            else:
                ax, qkv, attn = attn(x, tar_layer=None)

            qkvs.append(qkv)
            attns.append(attn)
            # print('ax: ', ax.shape)  # torch.Size([8, 256, 256])
            x = ax + x
            x = ff(x) + x
            # print('x: ', x.shape)  # torch.Size([8, 256, 256])

            count_layer += 1

            fea.append(self.recover1_patch_embedding(x))

        if tar_layer is not None:
            # print('Transformer_HP 2 tar_layer: ', tar_layer)
            return x, qkvs, attns, fea, tar_attn
        else:
            return x, qkvs, attns, fea


class TransformerDown_HP(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=4, dmodel=1024, mlp_dim=2048, patch_size=1, heads=6, dim_head=256, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_HP(self.dmodel, depth, heads, dim_head, image_height, patch_height, self.mlp_dim, dropout, image_height//patch_height, image_width//patch_width)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x, tar_layer=None, tmp_score=None):
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        #x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        if tar_layer is not None:
            ax, qkvs, attns, fea, tar_attn = self.transformer(x, tar_layer=tar_layer, tmp_score=tmp_score)
        else:
            ax, qkvs, attns, fea = self.transformer(x, tar_layer=tar_layer, tmp_score=tmp_score)
        out = self.recover_patch_embedding(ax)

        if tar_layer is not None:
            return out, qkvs, attns, fea, tar_attn
        else:
            return out, qkvs, attns, fea
