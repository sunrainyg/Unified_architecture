import torch
import torchvision
import torchvision.transforms as transforms
import trainer as t
from torch.utils.data import Dataset
import random
from skimage.draw import line
import torch.backends.cudnn as cudnn
import rfm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm
from random import randint
import visdom
import matplotlib.pyplot as plt
import hickle
import neural_model

SEED = 5636
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def to_one_hot(label, num_classes=2):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


def add_colored_blocks_and_save(subset, label_1, label_2):
    adjusted = []
    count_label_1 = 0
    count_label_2 = 0
    save_count_label_1 = 0
    save_count_label_2 = 0
    
    for idx, (ex, label) in enumerate(subset):
        if label == label_1:
            count_label_1 += 1
            # 为 label_1 添加红色方块
            ex[0, 1:3, 2:4] = 1.0  # Red channel
            ex[1, 1:3, 2:4] = 1.0  # Green channel
            ex[2, 1:3, 2:4] = 0.0  # Blue channel

            # 保存label_1的图片
            if save_count_label_1 < 5:
                plt.imshow(np.transpose(ex, (1, 2, 0)))
                filename = f"imgs/img_logs/label_{label_1}_image_{save_count_label_1}.png"
                plt.savefig(filename)
                print(f"Image saved as {filename}")
                plt.show()
                save_count_label_1 += 1

        elif label == label_2:
            count_label_2 += 1
            # 为 label_2 添加蓝色方块
            ex[0, 1:3, 2:4] = 0.0  # Red channel
            ex[1, 1:3, 2:4] = 0.0  # Green channel
            ex[2, 1:3, 2:4] = 0.0  # Blue channel

            # 保存label_2的图片
            if save_count_label_2 < 5:
                plt.imshow(np.transpose(ex, (1, 2, 0)))
                filename = f"imgs/img_logs/label_{label_2}_image_{save_count_label_2}.png"
                plt.savefig(filename)
                print(f"Image saved as {filename}")
                plt.show()
                save_count_label_2 += 1
        
        ex = ex.flatten()
        one_hot_label = to_one_hot(label)
        adjusted.append((ex, one_hot_label))

    return adjusted




def select_samples_from_cifar10(dataset, num_samples_1, num_samples_2, label_1, label_2):
    # 收集两个类别的样本
    samples_label_1 = [ex for ex in dataset if ex[1] == label_1]
    samples_label_2 = [ex for ex in dataset if ex[1] == label_2]
    
    # 打印两个类别在原数据集中的样本数量
    print(f"Total samples for label {label_1}: {len(samples_label_1)}")
    print(f"Total samples for label {label_2}: {len(samples_label_2)}")
    
    # 从每个类别的样本列表中选取指定数量的样本
    selected_samples_label_1 = samples_label_1[:num_samples_1]
    selected_samples_label_2 = samples_label_2[:num_samples_2]
    
    # 打印选取的样本数量
    print(f"Selected samples for label {label_1}: {len(selected_samples_label_1)}")
    print(f"Selected samples for label {label_2}: {len(selected_samples_label_2)}")
    
    # 合并两个类别的选定样本
    final_subset = selected_samples_label_1 + selected_samples_label_2
    
    # 打印最终选取的总样本数量
    print(f"Total selected samples: {len(final_subset)}")
    
    return final_subset



def one_hot_data_bak(dataset, num_samples=-1):
    '''
    subset: category 0 - 500 images; category 9 - 53 images
    '''
    # labelset = {}
    # for i in range(10):
    #     one_hot = torch.zeros(10)
    #     one_hot[i] = 1
    #     labelset[i] = one_hot
    label_1 = 0
    label_2 = 1
    
    labelset = {label_1: torch.tensor([1., 0.]), label_2: torch.tensor([0., 1.])}

    subset = [(ex, label) for idx, (ex, label) in enumerate(dataset) \
              if (idx < num_samples or num_samples == -1) and (label == label_1 or label == label_2)]


    adjusted = []

    count = 0
    
    count_label_1 = 0
    count_label_2 = 0
    
    for idx, (ex, label) in enumerate(subset):
        ex[:, 2:10, 7:15] = 0.
        if label == label_1:
            count_label_1 += 1
            ex[0, 2:10, 7:15] = 1.0
            ex[1, 2:10, 7:15] = 1.0
        elif label == label_2:
            count_label_2 += 1
            
        if idx < 10:
            plt.imshow(np.transpose(ex, (1, 2, 0)))
            filename = f"imgs/img_logs/image_{idx}.png"
            plt.savefig(filename)
            print(f"Image saved as {filename}")
            plt.show()
        ex = ex.flatten()
        adjusted.append((ex, labelset[label]))

    return adjusted



def one_hot_data(dataset, num_samples_1, num_samples_2):

    subset = select_samples_from_cifar10(dataset, num_samples_1, num_samples_2, label_1=0, label_2=1)
    adjusted = add_colored_blocks_and_save(subset, label_1=0, label_2=1)

    return adjusted

def split(trainset, p=.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val

def load_from_net(SIZE=64, path='./nn_models/trained_nn.pth'):
    dim = 3 * SIZE * SIZE
    net = neural_model.Net(dim, num_classes=10)

    d = torch.load(path)
    net.load_state_dict(d['state_dict'])
    for idx, p in enumerate(net.parameters()):
        if idx == 0:
            M = p.data.numpy()

    M = M.T @ M
    return M

def main():

    cudnn.benchmark = True
    global SIZE

    transform = transforms.Compose(
        [transforms.ToTensor()
        ])

    path = '~/datasets/'
    trainset = torchvision.datasets.CIFAR10(root=path,
                                          train=True,
                                          transform=transform,
                                          download=True)

    train_num_samples_1 = 250
    train_num_samples_2 = 250
    trainset = one_hot_data(trainset, train_num_samples_1, train_num_samples_2)
    print("Label shape:", trainset[0][1].shape) # Label shape: torch.Size([10])
    
    unique_labels = {}

    for _, label in trainset:
        label_str = str(label.numpy())
        if label_str not in unique_labels:
            unique_labels[label_str] = label

    # 打印所有独特的标签
    for key, value in unique_labels.items():
        print("unique labels:", key, value)
    
    trainset, valset = split(trainset, p=.8)

    print("Train Size: ", len(trainset), "Val Size: ", len(valset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=88,
                                              shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=88,
                                            shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path,
                                         train=False,
                                         transform=transform,
                                         download=True)
    testset = one_hot_data(testset, num_samples_1=50000, num_samples_2=50000)
    print(len(testset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                             shuffle=False, num_workers=2)


    name = '500 balance'
    # rfm.rfm(trainloader, valloader, testloader,
    #         name=name,
    #         iters=3,
    #         train_acc=True, reg=1e-3)

    t.train_network(trainloader, valloader, testloader,
                    name=name)


if __name__ == "__main__":
    main()
