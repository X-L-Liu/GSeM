import argparse
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from model_class import *


def eval_(attack):
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Test-Progress ', ncols=100) as pbar:
        for _, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image = attack(image, label)
            logit = model(image)
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def load_model():
    classifier = globals()[config.classifier_name](num_classes, config.alpha, config.beta)
    classifier.load_state_dict(torch.load(config.model_load_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    return classifier


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--classifier_name', type=str, default='GSeMResNet18')
    parser.add_argument('--model_load_path', type=str, default=r'')
    parser.add_argument('--alpha', type=int, default=6)
    parser.add_argument('--beta', type=int, default=4)
    parser.add_argument('--att_eot_num', type=int, default=20)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()

    assert os.path.exists(config.model_load_path)
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)

    device = torch.device(f'cuda:{config.device}')

    if config.dataset_name == 'cifar10':
        num_classes = 10
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=trans_cifar10_test)
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=trans_cifar100_test)
    else:
        num_classes = 10
        testSet = datasets.SVHN(root=config.data_path, split='test', download=True, transform=trans_svhn_test)

    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model = load_model()
    attacks = {
        'Clean': torchattacks.VANILA(model),
        'PGD500': AdaptivePGD(model, steps=500, eot_num=config.att_eot_num),
        'CW100': AdaptiveCW(model, steps=500, eot_num=config.att_eot_num),
        'APGD500': AdaptiveAPGD(model, steps=500, n_restarts=50, eot_num=config.att_eot_num),
        'AA': AdaptiveAutoAttack(model, eot_num=config.att_eot_num)
    }

    print('>' * 100)
    print(f'{config.classifier_name}  Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')

    for att_name in attacks.keys():
        torch.cuda.synchronize()
        acc = eval_(attacks[att_name])
        torch.cuda.synchronize()
        print(f'{att_name}: {acc * 100:.2f}%')
