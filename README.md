# Dark - a PyTorch like library written from scratch (for education)
![Logo](logo-small.png)

## Modules

1) **dark-1**
Core elements: graph nodes, auto-diff engine for scalars only.
Sample: quadratic 1D function optimization
   
2) **dark-2**
Support for tensor objects.
Sample: logistic regression on 2D data.
   
3) **dark-3**
Implementation of higher-level classes: Module, Linear and ReLU layers and BCEWithLogits / CrossEntropy loss. SGD optimizer.
Sample: MNIST classification (without dataloader)
   
4) **dark-4**
Implementation of Dataset, DataLoader and transformation classes.
Sample: FashionMNIST classification - in a PyTorch style!

5) **dark-5** (TODO)
Implementation of Conv2D and MaxPool layers
Sample: CIFAR10 classification using CNN.

6) **dark-6** (TODO)
GPU support via CuPy
Sample: cat/dog classification using Resent9 written from scratch


## How to run
To run samples go to a module folder and install a package by running:
```
pip install --editable .
```

E.g.
```
cd dark-1
pip install --editable .
code .
```

All required packages should be installed automatically.
Note: it is recommended to create an environment first.