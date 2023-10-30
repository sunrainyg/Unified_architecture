import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from main import Hyper_ViT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def visualize_centers_with_dim_reduction(rbf_net, method='tsne'):
    centers = rbf_net.centers.detach().cpu().numpy()

    if method == 'pca':
        reduced_data = PCA(n_components=2).fit_transform(centers)
    elif method == 'tsne':
        reduced_data = TSNE(n_components=2, perplexity=5).fit_transform(centers)

    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    for i, txt in enumerate(range(1, 12)): # Adding text labels 1 to 11 for the centers
        plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]))
    plt.title(f"{method.upper()} Visualization of RBF Centers")
    plt.show()
    save_path='./rbf_centers_new.png'
    plt.savefig(save_path)
    plt.close()


def vis_center_class(model, rbf_net, save_path='./center_class4.png'):
    centers = rbf_net.centers.detach().cpu().numpy()
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='/om/user/yulu_gan/data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=50000, shuffle=True, num_workers=2)
    images, labels = next(iter(dataloader))

    # 为每个RBF中心确定主要类别
    outputs = model(images)
    print("outputs:", outputs.shape) #（5,10）- (b,cls)
    mean_activations_per_class = []
    for i in range(11):
        class_indices = (labels == i).nonzero(as_tuple=True)[0]
        mean_activations_per_class.append(outputs[class_indices].mean(dim=0))
    mean_activations_per_class = torch.stack(mean_activations_per_class)
    mean_activations_per_class = mean_activations_per_class[:10,:]
    print("mean_activations_per_class", mean_activations_per_class.shape)

    center_classes = torch.argmax(mean_activations_per_class, dim=0).numpy()
    print("center_classes:", center_classes)
    reduced_centers = PCA(n_components=2).fit_transform(centers)
    
    plt.scatter(reduced_centers[:10, 0], reduced_centers[:10, 1], c=center_classes, cmap='jet', s=100, edgecolors='k')
    plt.colorbar()
    plt.title("RBF Centers in 2D space")
    plt.savefig(save_path)
    plt.close()


    





def visualize_centers(rbf_net, save_path='./rbf_centers.png'):
    centers = rbf_net.centers.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(centers[:, 0], centers[:, 1])
    plt.title("RBF Centers")
    plt.savefig(save_path)
    plt.close()

def visualize_beta(rbf_net, save_path='./beta_over_epochs.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(rbf_net.beta_mean_history)
    plt.title("Beta mean over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Beta mean")
    plt.savefig(save_path)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from scipy.ndimage import zoom




def visualize_rbf_activations(model, num_samples=5, save_path='./rbf_activations_with_images.png'):
    # 1. Load a few CIFAR10 samples
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='/om/user/yulu_gan/data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=num_samples, shuffle=True, num_workers=2)
    images, labels = next(iter(dataloader))
    
    # 2. Get the output from the last RBF layer
    rbf_outputs = []
    def hook_fn(module, input, output):
        rbf_outputs.append(output)
    
    # Register hook
    rbf_layers = model.get_all_rbf_layers()
    hook = rbf_layers[-1].register_forward_hook(hook_fn)
    
    # 3. Forward pass through the model
    model(images)
    
    # 4. Remove the hook
    hook.remove()
    
    # Check if there are outputs in rbf_outputs
    assert len(rbf_outputs) == 1
    activations = rbf_outputs[0].detach().cpu().numpy()
    
    # 5. Visualize the activations with images
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i+1)
        plt.imshow(TF.to_pil_image(images[i].cpu()))
        plt.title(f"Sample {i+1}")
        plt.axis('off')
        
        plt.subplot(2, num_samples, num_samples+i+1)
        plt.plot(activations[i])
        plt.tight_layout()
    
    plt.suptitle("Original CIFAR10 samples with their Last RBF Activations")
    plt.savefig(save_path)
    plt.close()






# Load your trained RBFNetwork
model_path = "/om/user/yulu_gan/model/hyperbf_cifar10_best_depth4_head8_patch4.pth"

model = Hyper_ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 4,
    heads = 8,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)

state_dict = torch.load(model_path)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

print(model)
model.eval()

# Visualize
rbf_layer = model.get_all_rbf_layers()[3]
save_path = './center_class_4deep_4.png'
vis_center_class(model, rbf_layer, save_path)
# visualize_centers_with_dim_reduction(rbf_layer)
# visualize_beta(rbf_layer)
# visualize_rbf_activations(model)
