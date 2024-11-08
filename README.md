<!-- # &#x1F309; Language vison interface -->

&#x1F31F; Official PyTorch implementation of Unified Architectures. 

The master branch works with **PyTorch 1.5+**.

## Overview
We discovered a unified architectures that share similar performance with Transformers

[![pP6cvZT.png](https://s1.ax1x.com/2023/09/09/pP6cvZT.png)](https://imgse.com/i/pP6cvZT)

<details open>
<summary>Major features</summary>


## Links
* [Project Page](https://github.com) 
* [ColabDemo](https://colab.research.google.com/)
* [Paper](https://cbmm.mit.edu/sites/default/files/publications/CBMM-Memo-143.pdf)


## Set up the environments
Install dependencies by running:
```shell
conda env create -f environment.yaml
conda activate uniarc
```


## Get Started
```shell
python main.py --dataset cifar10
```


## Demo

* Run the demo on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)


## Acknowledgement

Code is partly based on [Vision Transformer](https://github.com/).


## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@techreport{gan2023homogeneous,
  title={A Homogeneous Transformer Architecture},
  author={Gan, Yulu and Poggio, Tomaso},
  year={2023},
  institution={Center for Brains, Minds and Machines (CBMM)}
}
```