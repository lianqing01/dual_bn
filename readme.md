# Code for Constraint Batch Normalization, details of the normalization is in models/constraint_bn_v2.py


## Setup
  1. Install the logger wandb (register in http://wandb.ai/):
    
    ```
    
    pip install wandb
    wandb login $YourID
   
    ```
    
## Running CIFAR10:

```
     sh config/cifar100/constraint.sh vgg16_mybn cbn 128 10 0.1 CIFAR10 --constraint_weighted_average False --optimal_multiplier True --grad_clip 1
```
The details of tuning the paramter is in config/cifar100/constraint.sh
