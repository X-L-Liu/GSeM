# Gradient Semi-Masking for Improving Adversarial Robustness

[//]: # ([Paper]&#40;&#41; )

> **Abstract:** In gradient masking, certain complex signal processing and probabilistic optimization strategies exhibit 
> favorable characteristics such as nonlinearity, irreversibility, and feature preservation, thereby providing new 
> solutions for adversarial defense. Inspired by this, this paper proposes a plug-and-play _gradient semi-masking module_ 
> (**GSeM**) to improve the adversarial robustness of neural networks. GSeM primarily contains a feature straight-through 
> pathway that allows for normal gradient propagation, and a feature mapping pathway that interrupts gradient flow. The 
> multi-pathway and semi-masking characteristics cause GSeM to exhibit opposing behaviors when processing data and 
> gradients. Specifically, during data processing, GSeM compresses the state space of features while introducing white 
> noise augmentation. However, during gradient processing, it leads to inefficient updates to certain parameters and 
> ineffective generation of training examples. To address this shortcoming, we correct gradient propagation and 
> introduce gradient-corrected adversarial training. Extensive experiments demonstrate that GSeM differs fundamentally 
> from earlier gradient masking methods: it can genuinely enhance the adversarial defense performance of neural 
> networks, surpassing previous state-of-the-art approaches.

## Installation

```
conda create -n GSeM python=3.11.9
conda activate GSeM
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
PyTorch is not version-sensitive. The project can typically run on other versions of PyTorch as well. 
Furthermore, allow the system to automatically select the version when installing any other missing libraries.

## Training GSeM

You can train the backbone model integrated with GSeM using "gradient-corrected adversarial training."

* For GSeM-ResNet-18 on CIFAR-10
  `python train_gsem.py --classifier_name GSeMResNet18 --dataset_name cifar10`

* For GSeM-ResNet-18 on CIFAR-100
  `python train_gsem.py --classifier_name GSeMResNet18 --dataset_name cifar100`

* For GSeM-ResNet-18 on SVHN
  `python train_gsem.py --classifier_name GSeMResNet18 --dataset_name svhn`

* For GSeM-WideResNet-28-10 on CIFAR-10
  `python train_gsem.py --classifier_name GSeMWideResNet28x10 --dataset_name cifar10`

* For GSeM-WideResNet-28-10 on CIFAR-100
  `python train_gsem.py --classifier_name GSeMWideResNet28x10 --dataset_name cifar100`

* For GSeM-WideResNet-28-10 on SVHN
  `python train_gsem.py --classifier_name GSeMWideResNet28x10 --dataset_name svhn`

## Training GSeM-MART

You can train the backbone model integrated with GSeM using "gradient-corrected adversarial training" & MART.

* For GSeM-ResNet-18 on CIFAR-10
  `python train_gsem_mart.py --classifier_name GSeMResNet18 --dataset_name cifar10`

* For GSeM-ResNet-18 on CIFAR-100
  `python train_gsem_mart.py --classifier_name GSeMResNet18 --dataset_name cifar100`

* For GSeM-ResNet-18 on SVHN
  `python train_gsem_mart.py --classifier_name GSeMResNet18 --dataset_name svhn`

* For GSeM-WideResNet-28-10 on CIFAR-10
  `python train_gsem_mart.py --classifier_name GSeMWideResNet28x10 --dataset_name cifar10`

* For GSeM-WideResNet-28-10 on CIFAR-100
  `python train_gsem_mart.py --classifier_name GSeMWideResNet28x10 --dataset_name cifar100`

* For GSeM-WideResNet-28-10 on SVHN
  `python train_gsem_mart.py --classifier_name GSeMWideResNet28x10 --dataset_name svhn`

**MART** in the paper 'Improving Adversarial Robustness Requires Revisiting Misclassified Examples' 
(https://openreview.net/forum?id=rklOg6EFwS)

## Training GSeM-AWP

You can train the backbone model integrated with GSeM using "gradient-corrected adversarial training" & AWP.

* For GSeM-ResNet-18 on CIFAR-10
  `python train_gsem_awp.py --classifier_name GSeMResNet18 --dataset_name cifar10`

* For GSeM-ResNet-18 on CIFAR-100
  `python train_gsem_awp.py --classifier_name GSeMResNet18 --dataset_name cifar100`

* For GSeM-ResNet-18 on SVHN
  `python train_gsem_awp.py --classifier_name GSeMResNet18 --dataset_name svhn`

* For GSeM-WideResNet-28-10 on CIFAR-10
  `python train_gsem_awp.py --classifier_name GSeMWideResNet28x10 --dataset_name cifar10`

* For GSeM-WideResNet-28-10 on CIFAR-100
  `python train_gsem_awp.py --classifier_name GSeMWideResNet28x10 --dataset_name cifar100`

* For GSeM-WideResNet-28-10 on SVHN
  `python train_gsem_awp.py --classifier_name GSeMWideResNet28x10 --dataset_name svhn`

**AWP** in the paper 'Adversarial Weight Perturbation Helps Robust Generalization' 
(https://proceedings.neurips.cc/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html)

## Evaluating GSeM or GSeM-MART or GSeM-AWP

You can evaluate the GSeM-ResNet-18 model trained on the CIFAR-10 dataset using methods such as "Clean," "PGD500," "C&W100," "APGD500," and "AutoAttack."

The invocation method is as follows:  
  `python evaluate_gsem.py --classifier_name GSeMResNet18 --dataset_name cifar10 --model_load_path xxxx.pt`
