import argparse
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from model_class import *


def hook_t_rdim(grad):
    global grad_t_rdim
    grad_t_rdim = grad


def hook_t0(grad):
    global grad_t_rdim
    return grad + grad_t_rdim


def train_epoch():
    model.train()
    top1 = AverageMeter()
    with tqdm(total=len(train_loader), desc='Train-Progress', ncols=100) as pbar:
        for _, (image, label) in enumerate(train_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image_adv = adp_pgd(image, label)
            awp = awp_adversary.calc_awp(inputs_adv=image_adv, targets=label)
            awp_adversary.perturb(awp)
            optimizer.zero_grad()
            for _ in range(config.gra_eot_num):
                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        logits_adv = model(image_adv)
                        adv_probs = F.softmax(logits_adv, dim=1)
                        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                        new_y = torch.where(tmp1[:, -1] == label, tmp1[:, -2], tmp1[:, -1])
                        loss_adv = F.cross_entropy(logits_adv, label) + F.nll_loss(
                            torch.log(1.0001 - adv_probs + 1e-12), new_y)
                        nat_probs = F.softmax(model(image), dim=1)
                        true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()
                        loss_robust = (1 / len(image)) * torch.sum(torch.sum(F.kl_div(
                            torch.log(adv_probs + 1e-12), nat_probs, reduction='none'), dim=1) *
                                                                   (1.0000001 - true_probs))
                        loss = loss_adv + 5 * loss_robust
                    hook_handle_1 = model.gsem_module.t0.register_hook(hook_t0)
                    hook_handle_2 = model.gsem_module.t_rdim.register_hook(hook_t_rdim)
                    scaler.scale(loss).backward()
                else:
                    logits_adv = model(image_adv)
                    adv_probs = F.softmax(logits_adv, dim=1)
                    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                    new_y = torch.where(tmp1[:, -1] == label, tmp1[:, -2], tmp1[:, -1])
                    loss_adv = F.cross_entropy(logits_adv, label) + F.nll_loss(
                        torch.log(1.0001 - adv_probs + 1e-12), new_y)
                    nat_probs = F.softmax(model(image), dim=1)
                    true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()
                    loss_robust = (1 / len(image)) * torch.sum(torch.sum(F.kl_div(
                        torch.log(adv_probs + 1e-12), nat_probs, reduction='none'), dim=1) * (1.0000001 - true_probs))
                    loss = loss_adv + 5 * loss_robust
                    hook_handle_1 = model.gsem_module.t0.register_hook(hook_t0)
                    hook_handle_2 = model.gsem_module.t_rdim.register_hook(hook_t_rdim)
                    loss.backward()
                hook_handle_1.remove()
                hook_handle_2.remove()
            for p in model.parameters():
                p.grad.data /= config.gra_eot_num
            if config.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            awp_adversary.restore(awp)
            top1.update((logits_adv.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def test_epoch():
    model.eval()
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Test-Progress ', ncols=100) as pbar:
        for _, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image[::2] = adp_pgd(image[::2], label[::2])
            logit = model(image)
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def main(Reload):
    global model_load_path
    for epoch in range(config.epochs):
        start = time.time()
        train_acc = train_epoch()
        test_acc = test_epoch()
        scheduler.step()
        if test_acc > config.best_acc:
            model_load_path = os.path.join(classifier_save_path,
                                           f'{config.classifier_name}_{config.alpha}_{config.beta}_{test_acc}.pt')
            torch.save(model.state_dict(), model_load_path)
            if os.path.exists(model_load_path.replace(str(test_acc), str(config.best_acc))):
                os.remove(model_load_path.replace(str(test_acc), str(config.best_acc)))
            config.best_acc = test_acc
        print(f'Model: {config.classifier_name}  '
              f'Reload: {Reload + 1}/{config.reload}  Epoch: {epoch + 1}/{config.epochs}  '
              f'Train-Top1: {train_acc * 100:.2f}%  Test-Top1: {test_acc * 100:.2f}%  '
              f'Best-Top1: {config.best_acc * 100:.2f}%  Time: {time.time() - start:.0f}s')


def load_model():
    classifier = globals()[config.classifier_name](num_classes, config.alpha, config.beta)
    if model_load_path != '':
        classifier.load_state_dict(torch.load(model_load_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    return classifier


def load_proxy():
    classifier = globals()[config.classifier_name](num_classes, config.alpha, config.beta)
    classifier.to(device)

    return classifier


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=3.5 * 1e-3)
    parser.add_argument('--milestones', type=tuple, default=(60, 110))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--reload', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--model_save_path', type=str, default=r'adv_model')
    parser.add_argument('--model_load_path', type=str, default=r'')
    parser.add_argument('--best_acc', type=float, default=0)
    parser.add_argument('--classifier_name', type=str, default='GSeMResNet18')
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--alpha', type=int, default=6)
    parser.add_argument('--beta', type=int, default=4)
    parser.add_argument('--gra_eot_num', type=int, default=10)
    parser.add_argument('--att_eot_num', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    seed_random(0)
    torch.backends.cudnn.benchmark = True
    config = parse_args()

    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)

    device = torch.device(f'cuda:{config.device}')
    model_load_path = config.model_load_path

    if config.dataset_name == 'cifar10':
        num_classes = 10
        trainSet = datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=trans_cifar10_train)
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=trans_cifar10_test)
        classifier_save_path = os.path.join(config.model_save_path, 'cifar10')
    elif config.dataset_name == 'cifar100':
        num_classes = 100
        trainSet = datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=trans_cifar100_train)
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=trans_cifar100_test)
        classifier_save_path = os.path.join(config.model_save_path, 'cifar100')
    else:
        num_classes = 10
        trainSet = datasets.SVHN(root=config.data_path, split='train', download=True, transform=trans_svhn_train)
        testSet = datasets.SVHN(root=config.data_path, split='test', download=True, transform=trans_svhn_test)
        classifier_save_path = os.path.join(config.model_save_path, 'svhn')
    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if not os.path.exists(classifier_save_path):
        os.makedirs(classifier_save_path)

    grad_t_rdim = torch.empty(1)

    for reload in range(config.reload):
        model = load_model()
        adp_pgd = AdaptivePGD(model, eot_num=config.att_eot_num, use_amp=config.use_amp)
        print('>' * 100)
        print(f'{config.classifier_name}  Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')
        optimizer = optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
        scaler = torch.cuda.amp.GradScaler()
        proxy = load_proxy()
        proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
        awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=0.01,
                                         eot_num=config.gra_eot_num, use_amp=config.use_amp)
        main(reload)
