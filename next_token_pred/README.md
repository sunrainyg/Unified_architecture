<img src="animation.gif" width="200px" />

# PixelCNN
PixelCNN on grayscale celeb/mnist/cifar 28x28.
```
python train.py --ds mnist
python train.py --ds cifar
python train.py --ds celeb # assumes celeb faces can be loaded from 'celeb.npz' 
```

Code is based on <a href="https://github.com/singh-hrituraj/PixelCNN-Pytorch" target="_blank">this repository</a>.

Changes
- uses command line arguments with argparse instead of config files
- added cifar/celeb as additional datasets to mnist. 
- integrated with tensorboard 
- wrote code for inpainting
- made the above animation of generation process (code for animation not included).  
- wrote test case for likelihood computations. 
- changed hyperparameter slightly (increased nlayers and ksize)
- simplified code and moved into a single file (besides dataloading that also has its own file). 
