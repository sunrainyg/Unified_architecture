import torch
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import ipywidgets as widgets
from argparse import ArgumentParser
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
warnings.filterwarnings("ignore")

import pdb


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# class RBFNetwork(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
        
#         self.M_RBF = torch.nn.Parameter(torch.eye(768), requires_grad=True)
#         self.centers = nn.Parameter(torch.randn(hidden_features, in_features))
#         self.beta = nn.Parameter(torch.ones(hidden_features) * 0.1)  # scale factor
#         self.L  = nn.Parameter(torch.randn(in_features, in_features))
        
#         self.fc1 = nn.Linear(hidden_features, hidden_features)
#         self.fc2 = nn.Linear(hidden_features, out_features)

#     def radial_function(self, x): #欧式距离
#         # Compute the distance from the centers
#         A = x.pow(2).sum(dim=-1, keepdim=True)
#         B = self.centers.pow(2).sum(dim=1)
#         C = 2 * x @ self.centers.t()
#         distances = A - C + B
#         return torch.exp(-self.beta.unsqueeze(0) * distances)

#     # def radial_function(self, x): #马氏距离
#     #     # Compute the Mahalanobis distance from the centers using the decomposition method

#     #     # 1. Compute x^T M x
#     #     x_M = x @ self.M_RBF
#     #     xMx = (x * x_M).sum(dim=-1, keepdim=True)  # Assuming x is of shape [batch_size, dim]

#     #     # 2. Compute centers^T M centers
#     #     centers_M = self.centers @ self.M_RBF
#     #     centersMcenters = (self.centers * centers_M).sum(dim=-1)  # Assuming centers is of shape [num_centers, dim]

#     #     # 3. Compute -2 x^T M centers
#     #     xMcenters = x_M @ self.centers.t()  # Assuming x is of shape [batch_size, dim] and centers is of shape [num_centers, dim]
#     #     negative_2xMcenters = -2 * xMcenters

#     #     # Combine all components to get Mahalanobis distance
#     #     distances = xMx + centersMcenters + negative_2xMcenters

#     #     return torch.exp(-self.beta.unsqueeze(0) * distances)


#     def forward(self, x):
#         x = self.radial_function(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class RBFNetwork(nn.Module):
    def __init__(self, in_features, center_feature=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        center_feature = center_feature or in_features
    
        self.centers = nn.Parameter(torch.randn(center_feature, in_features))
        self.beta = nn.Parameter(torch.ones(center_feature) * 0.001)  # scale factor
        self.fc = nn.Linear(center_feature, out_features, bias=False) # set bias as false
        self.drop = nn.Dropout(drop)

    def radial_function(self, x): #欧式距离
        # Compute the distance from the centers
        A = x.pow(2).sum(dim=-1, keepdim=True)
        B = self.centers.pow(2).sum(dim=1)
        C = 2 * x @ self.centers.t()
        distances = A - C + B
        print("self.beta in the rbf network", self.beta)
        return torch.exp(-self.beta.unsqueeze(0) * distances)
    
    # def radial_function(self, x): #拉普拉斯距离
    #     # Compute the distance from the centers
    #     A = x.sum(dim=-1, keepdim=True)
    #     B = self.centers.sum(dim=1)
    #     C = 2 * x @ self.centers.t()
    #     distances = A - C + B
    #     # return torch.exp(-self.beta.unsqueeze(0) * distances)
    #     return torch.exp(0.5 * distances)


    def forward(self, x):
        x = self.radial_function(x)
        x = self.fc(x)
        x = self.drop(x)
        print("self.beta", self.beta)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        hidden_features = int(20)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x): # x: ([256, 65, 768])
        x = self.fc1(x) # x: ([256, 65, 3072])
        x = self.act(x) # x: ([256, 65, 3072])
        # x = self.drop(x) # x: ([256, 65, 3072])
        x = self.fc2(x) # x: ([256, 65, 768])
        # x = self.drop(x) # x: ([256, 65, 768])
        return x


class Hyper_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        initial_scale = qk_scale if qk_scale is not None else head_dim ** -0.5
        self.scale = nn.Parameter(torch.tensor(initial_scale))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.M = torch.nn.Parameter(torch.eye(192), requires_grad=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        ## 内积版本
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        
        ## 欧式距离版本
        # distance_squared = ((q.unsqueeze(-2) - k.unsqueeze(-3))).sum(dim=-1)
        # attn = -distance_squared * self.scale
        
        ## 下面是上面这个的优化版，避免广播导致内存占用太大
        # q_norm = (q ** 2).sum(dim=-1, keepdim=True)  # B, num_heads, N, 1
        # k_norm = (k ** 2).sum(dim=-1, keepdim=True)  # B, num_heads, 1, N
        # dot_product = q @ k.transpose(-2, -1)  # B, num_heads, N, N
        # dists = q_norm + k_norm.transpose(-2, -1) - 2 * dot_product
        # attn = -dists * self.scale
        
        ## 马氏距离版本
        # 1. Compute q^T M q
    
        q_M = q @ self.M
        q1 = q
        qMq = (q * q_M).sum(dim=-1, keepdim=True)  # B, num_heads, N, 1
        # 2. Compute k^T M k
        k_M = k @ self.M
        kMk = (k * k_M).sum(dim=-1, keepdim=True).transpose(-2, -1)  # B, num_heads, 1, N
        # 3. Compute -2 q^T M k
        qMk = q_M @ k.transpose(-2, -1)  # B, num_heads, N, N
        negative_2qMk = -2 * qMk
        # Combine all components to get Mahalanobis distance
        dists = qMq + kMk + negative_2qMk
        attn = -dists * self.scale # I don't think we need to scale it
        # attn = -dists
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        print("self.scale in attn layer:", self.scale)
        return x, attn, self.M


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or head_dim ** -0.8
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.M = torch.nn.Parameter(torch.eye(192), requires_grad=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        ## 内积版本
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, self.M


class Hyper_Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Hyper_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(20)
        self.mlp = RBFNetwork(dim, mlp_hidden_dim, dim)

    def forward(self, x, return_attention=False):
        y, attn, Ma = self.attn(self.norm1(x))
        if return_attention:
            return attn, Ma
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn, Ma = self.attn(self.norm1(x))
        if return_attention:
            return attn, Ma
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Hyper_ViT(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=4,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        print("1")
        self.blocks = nn.ModuleList([
            Hyper_Block(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # torch.Size([256, 65, 768])
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_first_selfattention(self, x):
      x = self.prepare_tokens(x)
      for i, blk in enumerate(self.blocks):
            if i == 0:
            # return attention of the first block
              return blk(x, return_attention=True)
            x = blk(x)


    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class ViT(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=4,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier headf
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_first_selfattention(self, x):
      x = self.prepare_tokens(x)
      for i, blk in enumerate(self.blocks):
            if i == 0:
            # return attention of the first block
              return blk(x, return_attention=True)
            x = blk(x)


    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img


def visualize_predict(model, img, img_size, patch_size, device):
    img_pre = transform(img, img_size)
    attention, ma = visualize_attention(model, img_pre, patch_size, device)
    plot_attention(img, attention)
    return ma


def visualize_attention(model, img, patch_size, device):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions, ma = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # import pdb;pdb.set_trace()
    attentions = attentions.reshape(nh, w_featmap, h_featmap)

    if device == torch.device("cpu"):
        attentions = nn.functional.interpolate(attentions.unsqueeze(
            0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()
    else:
        attentions = nn.functional.interpolate(attentions.unsqueeze(
            0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions, ma


def plot_attention(img, attention):
    n_heads = attention.shape[0]

    # Compute rows and cols for subplots
    cols = 3
    rows = (n_heads + cols - 1) // cols  # This ensures enough rows to fit all heads

    plt.figure(figsize=(12, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        ax = plt.subplot(1, 2, i+1)
        cax = plt.imshow(fig, cmap='viridis')
        plt.title(text[i])
        plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)  # fraction and pad help adjust the size and position of colorbar
    plt.savefig('attention_1.png')  # Save the figure before calling show

    plt.figure(figsize=(12, 10))
    for i in range(n_heads):
        ax = plt.subplot(rows, cols, i+1)
        cax = plt.imshow(attention[i], cmap='viridis')
        plt.title(f"Head n: {i+1}")
        plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
    plt.tight_layout()
    plt.savefig('attention_2.png')  # Save the figure before calling show


class Loader(object):
    def __init__(self):
        self.uploader = widgets.FileUpload(accept='image/*', multiple=False)
        self._start()

    def _start(self):
        display(self.uploader)

    def getLastImage(self):
        try:
            for uploaded_filename in self.uploader.value:
                uploaded_filename = uploaded_filename
            img = Image.open(io.BytesIO(
                bytes(self.uploader.value[uploaded_filename]['content'])))

            return img
        except:
            return None

    def saveImage(self, path):
        with open(path, 'wb') as output_file:
            for uploaded_filename in self.uploader.value:
                content = self.uploader.value[uploaded_filename]['content']
                output_file.write(content)


def visualize_and_save_matrix(M, filename="matrix_heatmap.png"):
    plt.figure(figsize=(10, 10))  # Adjust the figure size as necessary
    
    # Display the heatmap
    im = plt.imshow(M, cmap="viridis", aspect="auto")  # Choose an appropriate colormap, 'viridis' is just one example
    
    # Add a colorbar
    plt.colorbar(im)
    
    # Optional: Add title and labels if needed
    plt.title("Heatmap of Matrix M")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate(loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.*correct/total


def train(epochs):
    global best_acc
    
    for epoch in range(epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1) # outputs: torch.Size([256, 768])
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
               
        train_acc = 100.*correct/total
        val_acc   = evaluate(valloader)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '/om2/group/cbmm/data/best_model.pth')
            
        if args.wandb:
            wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss/(batch_idx+1),
                "Train Acc": train_acc,
                "Val Acc": val_acc
            })
                
        print('epoch:', epoch)
        print('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Val Acc: %.3f%%' % val_acc)
        print('Best Acc: %.3f%%' % best_acc)

def print_parameters_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


class TinyImageNet(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.data = []
        self.targets = []

        wnid_to_idx = {}
        with open(os.path.join(root, 'wnids.txt'), 'r') as f:
            for idx, line in enumerate(f):
                wnid_to_idx[line.strip()] = idx

        if split == 'train':
            for wnid, idx in wnid_to_idx.items():
                for i in range(500):
                    path = os.path.join(root, 'train', wnid, 'images', '{}_{}.JPEG'.format(wnid, i))
                    self.data.append(path)
                    self.targets.append(idx)
        elif split == 'val':
            with open(os.path.join(root, 'val', 'val_annotations.txt'), 'r') as f:
                for line in f:
                    parts = line.split()
                    path = os.path.join(root, 'val', 'images', parts[0])
                    self.data.append(path)
                    self.targets.append(wnid_to_idx[parts[1]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        target = self.targets[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='cifar10', type=str)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--vis", action='store_true', default=False, help='if doing visualization of attn. weights')
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--hyperbf", action='store_true', default=False, help='if using all hyperBF structures')
    parser.add_argument("--wandb", action='store_true', default=False, help='if using wandb')
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project="vit_new")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if device.type == "cuda":
        torch.cuda.set_device(0)


    ## Load datasets
    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

    transform_val = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
    
    transform_tinyimagenet_train = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(64, padding=4),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
    
    transform_tinyimagenet_val = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(64, padding=4),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

    if args.dataset == 'tiny_imagenet':
        train_dataset = TinyImageNet('/om2/group/cbmm/data', split='train', transform=transform_tinyimagenet_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        val_dataset = TinyImageNet('/om2/group/cbmm/data', split='val', transform=transform_tinyimagenet_val)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    if args.dataset == 'cifar10':
        trainset    = datasets.CIFAR10(root='/om2/group/cbmm/data', train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)
        valset      = datasets.CIFAR10(root='/om2/group/cbmm/data', train=False, download=True, transform=transform_val)
        valloader   = DataLoader(valset, batch_size=256, shuffle=False, num_workers=8)
        testset     = datasets.CIFAR10(root='/om2/group/cbmm/data', train=False, download=True, transform=transform_val)
        testloader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)
    
    if args.dataset == 'cifar100':
        trainset    = datasets.CIFAR100(root='/om2/group/cbmm/data', train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)
        valset      = datasets.CIFAR100(root='/om2/group/cbmm/data', train=False, download=True, transform=transform_val)
        valloader   = DataLoader(valset, batch_size=256, shuffle=False, num_workers=8)
        testset     = datasets.CIFAR100(root='/om2/group/cbmm/data', train=False, download=True, transform=transform_val)
        testloader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)
        

    ## load models
    if args.hyperbf:
        
        model = Hyper_ViT(
            image_size = 32,
            patch_size = args.patch_size,
            num_classes = 10,
            dim = 512,
            depth = 4,
            heads = 4,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
    else:
        model = ViT(
            image_size = 32,
            patch_size = args.patch_size,
            num_classes = 10,
            dim = 512,
            depth = 4,
            heads = 4,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
    print_parameters_count(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## setup loss, optimizer, etc
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    ## train
    train(args.epoch)
    
    test_acc = evaluate(testloader)
    print('Test Acc: %.3f%%' % test_acc)
    torch.cuda.empty_cache()
    
    if args.vis:
        path            = '/lustre/grp/gyqlab/lism/brt/language-vision-interface/N-W-estimator/vision-transformers-cifar10/corgi_image.jpg'
        img             = Image.open(path)
        factor_reduce   = 2
        img_size        = tuple(np.array(img.size[::-1]) // factor_reduce)
        model.to('cpu')
        device          =  torch.device("cpu")
        ma              = visualize_predict(model, img, img_size, args.path_size, device)
        visualize_and_save_matrix(ma.detach().cpu().numpy())
    
    if args.wandb:
        wandb.finish()
