"""
    Simple PixelCNN with softmax.

"""

import torch
import numpy as np 

from torch import nn
from torch.utils.tensorboard import SummaryWriter

import torchvision 
import torchvision.datasets  as datasets
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt

from load_data import load_data
import argparse 
from test_case import compute_nll

def compute_nll(output, images):
    """
        Naive implementation of nll computation, used for test case, very slow. 
        Think of output of nn as a table that perfectly describes the discrete model density.
        NLL is the negative sum of logs of the entries in this table that match the pixels in image. 
    """
    bs, v, h, w = output.shape
    images      = images*255 

    output = nn.functional.softmax(output, dim=1)

    nll = 0
    for i in range(bs): 
        for j in range(h):
            for k in range(w):
                pixel = int(images[i,0,j,k])
                nll  -= torch.log(output[i, pixel, j, k])

    return nll  / (bs*h*w*np.log(2))


class MaskedConv(nn.Conv2d):
	def __init__(self, mask_type, *args, **kwargs):
		super(MaskedConv, self).__init__(*args, **kwargs)
		self.mask_type = mask_type
		self.register_buffer('mask', self.weight.data.clone())

		channels, depth, height, width = self.weight.size()

		self.mask.fill_(1)
		if mask_type =='A':
			self.mask[:,:,height//2,width//2:] = 0
			self.mask[:,:,height//2+1:,:] = 0
		else: 
			self.mask[:,:,height//2,width//2+1:] = 0
			self.mask[:,:,height//2+1:,:] = 0


	def forward(self, x):
		self.weight.data *= self.mask 
		return super(MaskedConv, self).forward(x)


class PixelCNN(nn.Module):

    def __init__(self, args):
        super(PixelCNN, self).__init__()
        nlayers       = args.nlayers 
        ksize         = args.ksize 
        channels      = args.channels 
        self.nlayers  = nlayers

        self.Conv2d_1          = MaskedConv('A', 1, channels, ksize, 1, ksize//2, bias=False)
        self.BatchNorm2d_1     = nn.BatchNorm2d(channels)

        self.convs  = []
        self.bnorms = []
        for i in range(nlayers): 
            self.convs.append(MaskedConv('B', channels, channels, ksize, 1, ksize//2, bias=True))
            self.bnorms.append(nn.BatchNorm2d(channels))

        self.convs      = nn.ModuleList(self.convs)
        self.bnorms     = nn.ModuleList(self.bnorms)

        self.out = nn.Conv2d(channels, 256, 1) 

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.BatchNorm2d_1(x)
        x = nn.functional.relu(x)  

        for i in range(self.nlayers):  # residual layers. 
            x = self.bnorms[i](x) 
            y = self.convs[i](x) 
            x = nn.functional.relu(y+x)

        return self.out(x)


def visualize_weights(conv_layer):
    # 获取权重
    weights = conv_layer.weight.data.cpu().numpy()
    
    # 为简单起见，只展示第一个输出通道的权重
    # 因为每个输出通道可能关注于不同的特征
    weights_to_show = weights[0, 0, :, :]
    
    plt.imshow(weights_to_show, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.show()


def main(args):
    title = str(vars(args))
    print(title)
    writer  = SummaryWriter(comment=title)

    layers       = args.nlayers 
    kernel       = args.ksize 
    channels     = args.channels 
    epochs       = args.epochs 
    lr           = args.lr 
    batch_size   = args.bs


    trainloader  = load_data(args, args.ds) # supports mnist, cifar and celeba. 

    net          = PixelCNN(args).to('cuda')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # combines nllloss and logsoftmax in one. 

    net.train()

    global_step = 0 

    for epoch in range(epochs):

        for images in trainloader:
            # If MNIST then remove labels. 
            if type(images) == type([]): images = images[0] 
            images = images.cuda()

            optimizer.zero_grad()
            output     = net(images)
            loss       = criterion(output, (images[:, 0, :, :]*255).long())  / np.log(2)


            # Test case that our thing computes the same as cross entropy. 
            if global_step == 0: 
                nll        = compute_nll(output, images)
                assert torch.allclose(nll, loss, atol=0.01), (nll, loss)


            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/nll", loss, global_step=global_step)
            print("\r[%i / %i] nll=%.4f "%(global_step % len(trainloader), len(trainloader), loss.item()), end="", flush=True)

            if global_step % 100 == 0:
                net.eval()
                # Sample picture pixel by pixel. 
                sample = torch.zeros(10, 1, 28, 28, device='cuda')
                for i in range(28):
                    for j in range(28):
                        out             = net(sample)
                        probs           = nn.functional.softmax(out[:,:,i,j], dim=-1).data
                        sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
                writer.add_image("Image/sample", torchvision.utils.make_grid(sample), global_step=global_step)

                # Do inpainting pixel by pixel. 
                sample = torch.zeros(10, 1, 28, 28, device='cuda')
                sample[:, :, :14, :] = images[:10, :, :14, :]
                for i in range(14, 28): # start only after the real pixels. 
                    for j in range(28):
                        out             = net(sample)
                        probs           = nn.functional.softmax(out[:,:,i,j], dim=-1).data
                        sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
                writer.add_image("Image/inpaint", torchvision.utils.make_grid(sample), global_step=global_step)
                net.train()

            global_step  += 1


        print('Epoch: '+str(epoch)+' Over!')

        #Saving the model
        #os.makedirs('models', exist_ok=True)
        #torch.save(net.state_dict(), 'models/%i_%s.pth'%(epoch, title))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Arguments for PixelCNN')

    parser.add_argument('--nlayers',    default=7,       type=int,      help='Batch size')
    parser.add_argument('--ksize',      default=9,       type=int,      help='Kernel Size')
    parser.add_argument('--channels',   default=64,      type=int,      help='Channels')
    parser.add_argument('--bs',         default=128,     type=int,      help='Batch Size')
    parser.add_argument('--lr',         default=0.001,   type=float,    help='Learning Rate')
    parser.add_argument('--epochs',     default=100,     type=int,      help='Epochs')
    parser.add_argument('--ds',         default="mnist", type=str,      help='Dataset "mnist", "celeb" or "cifar".')

    args = parser.parse_args()

    main(args)
    
    # # 假设你已经有了一个训练过的PixelCNN模型，net
    # # 可视化第一层的权重
    # visualize_weights(net.Conv2d_1)

    # # 可视化其他层的权重（例如第2层）
    # visualize_weights(net.convs[1])




