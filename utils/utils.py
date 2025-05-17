from datetime import datetime
from typing import Callable, Optional
from torch import nn
from torchvision.datasets import VisionDataset
from torchvision import transforms
from .autoaugment import *
from .cutout import Cutout
import os
import pickle
import torch
import torchattacks


trans_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

trans_cifar10_train_resize64 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
])

trans_cifar10_test = transforms.Compose([
    transforms.ToTensor()
])

trans_cifar10_test_resize64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

trans_cifar100_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

trans_cifar100_train_resize64 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

trans_cifar100_test = transforms.Compose([
    transforms.ToTensor()
])

trans_cifar100_test_resize64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

trans_svhn_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

trans_svhn_train_resize64 = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])

trans_svhn_test = transforms.Compose([
    transforms.ToTensor()
])

trans_svhn_test_resize64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

trans_miniimagenet_train = transforms.Compose([
    transforms.RandomCrop(64, padding=8, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=24),
])

trans_miniimagenet_test = transforms.Compose([
    transforms.ToTensor()
])


class MiniImageNet(VisionDataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 pixel: str = '64',
                 transform: Optional[Callable] = None
                 ):
        super().__init__(root, transform=transform)
        data = pickle.load(file=open(os.path.join(self.root, 'mini-imagenet-' + pixel + '.pkl'), 'rb'))
        if train:
            self.sample, self.label = data['train_sample'], data['train_label']
        else:
            self.sample, self.label = data['test_sample'], data['test_label']
        self.sample = self.sample.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        img, target = self.sample[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.label)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.acc_num = 0
        self.total_num = 0
        self.acc_rate = 0

    def update(self, acc_num, total_num):
        self.acc_num += acc_num
        self.total_num += total_num
        self.acc_rate = self.acc_num / self.total_num


class SelfPrint:
    def __init__(self, print_name=None):
        if print_name is None:
            print_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_file = f'logs/{print_name}.txt'

    def __call__(self, info):
        print(info)
        with open(self.log_file, 'a') as file:
            print(info, file=file)


def seed_random(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AMP_PGD(torchattacks.PGD):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, use_amp=False):
        super().__init__(model, eps, alpha, steps, random_start)
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.get_logits(adv_images)

                    # Calculate loss
                    if self.targeted:
                        cost = -loss(outputs, target_labels)
                    else:
                        cost = loss(outputs, labels)
                    cost = self.scaler.scale(cost)
            else:
                outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
