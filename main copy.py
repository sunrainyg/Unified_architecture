import torch
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import misc as utils
import ipywidgets as widgets
from argparse import ArgumentParser
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
warnings.filterwarnings("ignore")


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

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RBFNetwork(nn.Module):
    def __init__(self, in_features, center_feature=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        center_feature = center_feature or in_features
        self.beta_mean_history = []
    
        self.centers = nn.Parameter(torch.randn(center_feature, in_features))
        self.beta = nn.Parameter(torch.ones(center_feature) * 0.01)
        self.fc = nn.Linear(center_feature, out_features)
        self.drop = nn.Dropout(drop)


    def radial_function(self, x):
        
        A = x.pow(2).sum(dim=-1, keepdim=True)
        B = self.centers.pow(2).sum(dim=1)
        C = 2 * x @ self.centers.t()
        distances = A - C + B
        
        current_beta_mean = self.beta.mean().item()
        self.beta_mean_history.append(current_beta_mean)
        
        return torch.exp(-self.beta.unsqueeze(0) * distances)


    def forward(self, x):
        x = self.radial_function(x)
        x = self.fc(x)
        x = self.drop(x)
        return x

    def save_epoch_value(self):
        current_beta_mean = self.beta.mean().item()
        self.beta_mean_history.append(current_beta_mean)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        hidden_features = int(1500)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Hyper_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scaler_mean_history = []
        
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
        
        distance_squared = ((q.unsqueeze(-2) - k.unsqueeze(-3))).sum(dim=-1)
        attn = -distance_squared * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def save_epoch_value(self):
        value_to_save = self.scale.mean().item()
        self.scaler_mean_history.append(value_to_save)


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
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        mlp_hidden_dim = int(1500)
        self.mlp = RBFNetwork(dim, mlp_hidden_dim, dim)

    def forward(self, x):
        y = self.attn(self.norm1(x))
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

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class PatchEmbed(nn.Module):
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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=4,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
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
        return self.head(x[:, 0])


class ViT(nn.Module):

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
        return self.head(x[:, 0])


def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img


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


def evaluate(loader):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        val_loss  = val_loss/(batch_idx+1)
        
    return 100.*correct/total, val_loss

def get_lr(optimizer):
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    return lrs

def train(model, epochs):
    global best_acc
    if 'best_acc' not in globals():
        best_acc = 0
    
    beta_mean_history_list      = []
    scaler_mean_history_list    = []
    for epoch in range(epochs):
        if epoch < 10:
            scheduler_warmup.step()
        
        current_lrs = get_lr(optimizer)
        print("Current Learning Rates:", current_lrs)
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print(f"Processing batch {batch_idx+1}/{len(trainloader)}", end="\r")

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
            del inputs, targets, outputs
            torch.cuda.empty_cache()
        
        train_acc = 100.*correct/total
        val_acc, val_loss   = evaluate(valloader)
        scheduler_plateau.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './model/best_model.pth')

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




if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, default='cifar10', type=str)
    parser.add_argument("--epoch", required=True, default=5, type=int)
    parser.add_argument("--vis", action='store_true', default=False, help='if doing visualization of attn. weights')
    parser.add_argument("--patch_size", required=True, default=4, type=int)
    parser.add_argument("--image_size", required=True, default=32, type=int)
    parser.add_argument("--hyperbf", action='store_true', default=False, help='if using all hyperBF structures')
    parser.add_argument("--wandb", action='store_true', default=True, help='if using wandb')
    parser.add_argument("--classes", required=True, default=10, type=int)
    parser.add_argument("--train_batch", required=True, default=256, type=int)
    parser.add_argument("--lr", required=True, default=1e-4, type=float)
    #Device options
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    
    if args.wandb:
        wandb.init(project="vit_new")
        
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))


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
    
    transform_imagenet = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

    if args.dataset == 'imagenet':
        print("loading the imagenet dataset")
        trainset = datasets.ImageFolder(root='./data', transform=transform_imagenet)
        valset = datasets.ImageFolder(root='./data', transform=transform_imagenet)
        train_sampler = DistributedSampler(trainset)
        val_sampler = DistributedSampler(valset)
        trainloader = DataLoader(trainset, batch_size=args.train_batch, sampler=train_sampler, num_workers=8)
        valloader = DataLoader(valset, batch_size=args.train_batch, sampler=val_sampler, num_workers=8)
    
    if args.dataset == 'cifar10':
        trainset    = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        valset      = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        testset     = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        
        train_sampler   = DistributedSampler(trainset)
        val_sampler     = DistributedSampler(valset)
        test_sampler     = DistributedSampler(testset)
        
        trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)
        valloader   = DataLoader(valset, batch_size=256, shuffle=False, num_workers=8)
        testloader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=8)
        

    ## load models
    if args.hyperbf:
        
        model = Hyper_ViT(
            image_size = args.image_size,
            patch_size = args.patch_size,
            num_classes = args.classes,
            dim = 512,
            depth = 4,
            heads = 4,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
    else:
        model = ViT(
            image_size = args.image_size,
            patch_size = args.patch_size,
            num_classes = args.classes,
            dim = 512,
            depth = 4,
            heads = 4,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    print("model loaded!!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model is on: {next(model.parameters()).device}")
    
    model       = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    ## setup loss, optimizer, etc
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / 10)
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    ## train
    beta, scaler = train(model, args.epoch)

    if args.dataset == 'cifar10' or 'cifar100':
        test_acc = evaluate(testloader)
        print('Test Acc: %.3f%%' % test_acc)
    if args.wandb:
        wandb.finish()
