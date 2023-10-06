import torch
import torchvision 
import numpy as np 
import torchvision.datasets  as datasets
import torchvision.transforms as transforms 

def load_data(args, name="mnist"):
    batch_size = args.bs 
    # ---------- load data.  ---------
    if name == "celeb": 
        from torch.utils.data import Dataset, DataLoader
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        X = torch.from_numpy(np.load("celeb.npz")['data']).float()
        X = X.view(-1, 1, 28, 28)
        trainloader = DataLoader(X, batch_size=batch_size, drop_last=True, shuffle=True)

    elif name == "mnist": 

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        mnist_train  = datasets.MNIST(root='./data', train=True,  download=True, transform=transform_train)
        mnist_test   = datasets.MNIST(root='./data', train=False, download=True, transform=None)

        trainloader  = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
        testloader   = torch.utils.data.DataLoader(mnist_test,  batch_size=batch_size, shuffle=False)

    elif name == "cifar": 

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(), 
            transforms.Resize((28, 28), interpolation=2),
            transforms.ToTensor()
        ])

        mnist_train  = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
        trainloader  = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)


    else: assert False, "Only accept 'mnist' or 'celeb' not %s. "%name
    # ---------- done loading data ----------

    return trainloader

