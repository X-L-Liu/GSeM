from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn as nn
import torchattacks
from torch import optim
from torchattacks import MultiAttack
from utils.corrected_fab import *


def hook_t_rdim(grad):
    global grad_t_rdim
    grad_t_rdim = grad


def hook_t0(grad):
    global grad_t_rdim
    return grad + grad_t_rdim


class GradSemiMaskModule(nn.Module):
    def __init__(self, in_planes, alpha, beta):
        super(GradSemiMaskModule, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_planes * 2, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_planes)
        self.t0 = torch.empty(1, requires_grad=True)
        self.t_rdim = torch.empty(1, requires_grad=True)

    def forward(self, t0):
        self.t0 = t0
        self.t_rdim = (torch.round(self.t0 * self.alpha) +
                       torch.randint_like(self.t0, -self.beta, self.beta + 1)) / self.alpha
        t_smth = self.conv1(self.t_rdim)
        t_cat = torch.cat((self.t0, t_smth), dim=1)
        t0_ = F.mish(self.bn(self.conv2(t_cat)))
        return t0_


class AdaptiveFGSM(torchattacks.FGSM):
    def __init__(self, model, eps=8 / 255, eot_num=20, use_amp=False):
        super().__init__(model, eps)
        self.eot_num = eot_num
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        global grad_t_rdim
        grad_t_rdim = torch.empty(1)

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        target_labels = self.get_target_label(images, labels) if self.targeted else labels

        loss = nn.CrossEntropyLoss()

        grad = torch.zeros_like(images)
        for _ in range(self.eot_num):
            images = images.detach()
            images.requires_grad = True
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.get_logits(images)
                    # Calculate loss
                    if self.targeted:
                        cost = -loss(outputs, target_labels)
                    else:
                        cost = loss(outputs, labels)
                cost = self.scaler.scale(cost)
            else:
                outputs = self.get_logits(images)
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
            # Update adversarial images

            hook_handle_1 = self.model.gsem_module.t0.register_hook(hook_t0)
            hook_handle_2 = self.model.gsem_module.t_rdim.register_hook(hook_t_rdim)
            grad += torch.autograd.grad(
                cost, images, retain_graph=False, create_graph=False
            )[0]
            hook_handle_1.remove()
            hook_handle_2.remove()

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


class AdaptivePGD(torchattacks.PGD):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, eot_num=20, use_amp=False):
        super().__init__(model, eps, alpha, steps, random_start)
        self.eot_num = eot_num
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        global grad_t_rdim
        grad_t_rdim = torch.empty(1)

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        target_labels = self.get_target_label(images, labels) if self.targeted else labels

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)
            for _ in range(self.eot_num):
                adv_images = adv_images.detach()
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

                hook_handle_1 = self.model.gsem_module.t0.register_hook(hook_t0)
                hook_handle_2 = self.model.gsem_module.t_rdim.register_hook(hook_t_rdim)
                grad += torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                hook_handle_1.remove()
                hook_handle_2.remove()

            adv_images = adv_images + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class AdaptiveCW(torchattacks.CW):
    def __init__(self, model, c=1, kappa=0, steps=50, lr=0.01, eot_num=20):
        super().__init__(model, c, kappa, steps, lr)
        self.eot_num = eot_num

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            cost_item = 0
            optimizer.zero_grad()
            for _ in range(self.eot_num):
                outputs = self.get_logits(adv_images)
                if self.targeted:
                    f_loss = self.f(outputs, target_labels).sum()
                else:
                    f_loss = self.f(outputs, labels).sum()
                cost = L2_loss + self.c * f_loss
                hook_handle_1 = self.model.gsem_module.t0.register_hook(hook_t0)
                hook_handle_2 = self.model.gsem_module.t_rdim.register_hook(hook_t_rdim)
                cost_item += cost.item()
                cost.backward(retain_graph=True)
                hook_handle_1.remove()
                hook_handle_2.remove()

            cost_item /= self.eot_num
            w.grad.data /= self.eot_num
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            if self.targeted:
                # We want to let pre == target_labels in a targeted attack
                condition = (pre == target_labels).float()
            else:
                # If the attack is not targeted we simply make these two values unequal
                condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost_item > prev_cost:
                    return best_adv_images
                prev_cost = cost_item

        return best_adv_images


class AdaptiveAutoAttack(Attack):
    def __init__(self, model, norm="Linf", eps=8 / 255, version="standard", n_classes=10, seed=None, verbose=False,
                 eot_num=20):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]

        if version == "standard":  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self._autoattack = MultiAttack([
                AdaptiveAPGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss="ce",
                             n_restarts=1, eot_num=eot_num),
                AdaptiveAPGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose,
                              n_classes=n_classes, n_restarts=1, eot_iter=eot_num),
                AdaptiveFAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, multi_targeted=True,
                             n_classes=n_classes, n_restarts=1, eot_num=eot_num),
                AdaptiveSquare(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000,
                               n_restarts=1),
            ])

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = MultiAttack([
                AdaptiveAPGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss="ce",
                             n_restarts=5, eot_num=eot_num),
                AdaptiveAPGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss="dlr",
                             n_restarts=5, eot_num=eot_num),
                AdaptiveFAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes,
                             n_restarts=5, eot_num=eot_num),
                AdaptiveSquare(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000,
                               n_restarts=1),
                AdaptiveAPGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose,
                              n_classes=n_classes, n_restarts=1, eot_iter=eot_num),
                AdaptiveFAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, multi_targeted=True,
                             n_classes=n_classes, n_restarts=1, eot_num=eot_num),
            ])

        elif version == "rand":  # ['apgd-ce', 'apgd-dlr']
            self._autoattack = MultiAttack([
                AdaptiveAPGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss="ce",
                             eot_num=20, n_restarts=1),
                AdaptiveAPGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss="dlr",
                             eot_num=20, n_restarts=1),
            ])
        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed


class AdaptiveAPGD(torchattacks.APGD):
    def __init__(self, model, norm="Linf", eps=8 / 255, steps=10, n_restarts=1, seed=0, loss="ce", eot_num=20,
                 rho=0.75, verbose=False):
        super().__init__(model, norm, eps, steps, n_restarts, seed, loss, eot_num, rho, verbose)
        self.steps_2 = None
        self.size_decr = None
        self.steps_min = None

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                        t.reshape([t.shape[0], -1])
                        .abs()
                        .max(dim=1, keepdim=True)[0]
                        .reshape([-1, 1, 1, 1])
                    )  # nopep8
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                            (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        if self.loss == "ce":
            criterion_indiv = nn.CrossEntropyLoss(reduction="none")
        elif self.loss == "dlr":
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError("unknown loss")

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                logits = self.get_logits(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
            grad += torch.autograd.grad(
                self.model.gsem_module.t0,
                x_adv,
                grad_outputs=torch.autograd.grad(loss, self.model.gsem_module.t_rdim)[0]
            )[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
                self.eps
                * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
                * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * grad / (
                            (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                                + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                                + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    logits = self.get_logits(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
                grad += torch.autograd.grad(
                    self.model.gsem_module.t0,
                    x_adv,
                    grad_outputs=torch.autograd.grad(loss, self.model.gsem_module.t_rdim)[0]
                )[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                    x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )  # nopep8
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                            loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv


class AdaptiveAPGDT(torchattacks.APGDT):
    def __init__(self, model, norm="Linf", eps=8 / 255, steps=10, n_restarts=1, seed=0, eot_iter=20, rho=0.75,
                 verbose=False, n_classes=10):
        super().__init__(model, norm, eps, steps, n_restarts, seed, eot_iter, rho, verbose, n_classes)
        self.size_decr = None
        self.steps_min = None
        self.steps_2 = None

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )  # nopep8
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )  # nopep8

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                        t.reshape([t.shape[0], -1])
                        .abs()
                        .max(dim=1, keepdim=True)[0]
                        .reshape([-1, 1, 1, 1])
                    )  # nopep8
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                            (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        output = self.get_logits(x)
        y_target = output.sort(dim=1)[1][:, -self.target_class]

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                logits = self.get_logits(x_adv)
                loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
            grad += torch.autograd.grad(
                self.model.gsem_module.t0,
                x_adv,
                grad_outputs=torch.autograd.grad(loss, self.model.gsem_module.t_rdim)[0]
            )[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
                self.eps
                * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
                * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        x_adv_old = x_adv.clone()
        # counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size[0] * grad / (
                            (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                                + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                                + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    logits = self.get_logits(x_adv)
                    loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
                grad += torch.autograd.grad(
                    self.model.gsem_module.t0,
                    x_adv,
                    grad_outputs=torch.autograd.grad(loss, self.model.gsem_module.t_rdim)[0]
                )[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                    x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )  # nopep8
                    fl_reduce_no_impr = (~reduced_last_check) * (
                            loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        # n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv


class AdaptiveFAB (CorrectedFAB):
    def __init__(self, model, norm="Linf", eps=8 / 255, steps=10, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9,
                 verbose=False, seed=0, multi_targeted=False, n_classes=10, eot_num=20):
        super().__init__(model, norm, eps, steps, n_restarts, alpha_max, eta, beta, verbose, seed, multi_targeted,
                         n_classes)
        self.eot_num = eot_num
        global grad_t_rdim
        grad_t_rdim = torch.empty(1)

    def get_diff_logits_grads_batch(self, imgs, la):
        df_t, dg_t = 0, 0
        im = imgs.clone()
        for _ in range(self.eot_num):
            im = im.detach().requires_grad_()

            with torch.enable_grad():
                y = self.get_logits(im)

            g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
            grad_mask = torch.zeros_like(y)
            for counter in range(y.shape[-1]):
                zero_gradients(im)
                grad_mask[:, counter] = 1.0
                hook_handle_1 = self.model.gsem_module.t0.register_hook(hook_t0)
                hook_handle_2 = self.model.gsem_module.t_rdim.register_hook(hook_t_rdim)
                y.backward(grad_mask, retain_graph=True)
                hook_handle_1.remove()
                hook_handle_2.remove()
                grad_mask[:, counter] = 0.0
                g2[counter] = im.grad.data

            g2 = torch.transpose(g2, 0, 1).detach()
            # y2 = self.get_logits(imgs).detach()
            y2 = y.detach()
            df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
            dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
            df[torch.arange(imgs.shape[0]), la] = 1e10

            df_t += df
            dg_t += dg

        df = df_t / self.eot_num
        dg = dg_t / self.eot_num

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        df_t, dg_t = 0, 0
        u = torch.arange(imgs.shape[0])
        im = imgs.clone()
        for _ in range(self.eot_num):
            im = im.detach().requires_grad_()

            with torch.enable_grad():
                y = self.get_logits(im)
                diffy = -(y[u, la] - y[u, la_target])
                sumdiffy = diffy.sum()

            zero_gradients(im)
            hook_handle_1 = self.model.gsem_module.t0.register_hook(hook_t0)
            hook_handle_2 = self.model.gsem_module.t_rdim.register_hook(hook_t_rdim)
            sumdiffy.backward()
            hook_handle_1.remove()
            hook_handle_2.remove()
            graddiffy = im.grad.data
            df = diffy.detach().unsqueeze(1)
            dg = graddiffy.unsqueeze(1)

            df_t += df
            dg_t += dg

        df = df_t / self.eot_num
        dg = dg_t / self.eot_num

        return df, dg


AdaptiveSquare = torchattacks.Square


class AdaptiveEADL1(torchattacks.EADL1):
    def __init__( self, model, kappa=0, lr=0.01, binary_search_steps=9, max_iterations=100, abort_early=True,
                  initial_const=0.001, beta=0.001, eot_num=20):
        super().__init__(model, kappa, lr, binary_search_steps, max_iterations, abort_early, initial_const, beta)
        self.eot_num = eot_num
        global grad_t_rdim
        grad_t_rdim = torch.empty(1)

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        outputs = self.get_logits(images)

        batch_size = images.shape[0]
        lower_bound = torch.zeros(batch_size, device=self.device)
        const = torch.ones(batch_size, device=self.device) * self.initial_const
        upper_bound = torch.ones(batch_size, device=self.device) * 1e10

        final_adv_images = images.clone()
        y_one_hot = torch.eye(outputs.shape[1]).to(self.device)[labels]

        o_bestl1 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestl1 = torch.Tensor(o_bestl1).to(self.device)
        o_bestscore = torch.Tensor(o_bestscore).to(self.device)

        # Initialization: x^{(0)} = y^{(0)} = x_0 in paper Algorithm 1 part
        x_k = images.clone().detach()
        y_k = nn.Parameter(images)

        # Start binary search
        for outer_step in range(self.binary_search_steps):

            self.global_step = 0

            bestl1 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            bestl1 = torch.Tensor(bestl1).to(self.device)
            bestscore = torch.Tensor(bestscore).to(self.device)
            prevloss = 1e6

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                const = upper_bound

            lr = self.lr
            for iteration in range(self.max_iterations):
                # reset gradient
                if y_k.grad is not None:
                    y_k.grad.detach_()
                    y_k.grad.zero_()

                for _ in range(self.eot_num):
                    # Loss over images_parameters with only L2 same as CW
                    # we don't update L1 loss with SGD because we use ISTA
                    output = self.get_logits(y_k)
                    L2_loss = self.L2_loss(y_k, images)

                    cost = self.EAD_loss(output, y_one_hot, None, L2_loss, const)
                    # cost.backward(retain_graph=True)

                    hook_handle_1 = self.model.gsem_module.t0.register_hook(hook_t0)
                    hook_handle_2 = self.model.gsem_module.t_rdim.register_hook(hook_t_rdim)
                    cost.backward()
                    hook_handle_1.remove()
                    hook_handle_2.remove()

                y_k.grad.data /= self.eot_num

                # Gradient step
                # y_k.data.add_(-lr, y_k.grad.data)
                self.global_step += 1
                with torch.no_grad():
                    y_k -= y_k.grad * lr

                # Ploynomial decay of learning rate
                lr = (
                    self.lr * (1 - self.global_step / self.max_iterations) ** 0.5
                )  # nopep8
                x_k, y_k = self.FISTA(images, x_k, y_k)
                # Loss ElasticNet or L1 over x_k
                with torch.no_grad():
                    output = self.get_logits(x_k)
                    L2_loss = self.L2_loss(x_k, images)
                    L1_loss = self.L1_loss(x_k, images)
                    loss = self.EAD_loss(
                        output, y_one_hot, L1_loss, L2_loss, const
                    )  # nopep8

                    # print('loss: {}, prevloss: {}'.format(loss, prevloss))
                    if (
                        self.abort_early
                        and iteration % (self.max_iterations // 10) == 0
                    ):
                        if loss > prevloss * 0.999999:
                            break
                        prevloss = loss

                    # L1 attack key step!
                    cost = L1_loss
                    self.adjust_best_result(
                        x_k,
                        labels,
                        output,
                        cost,
                        bestl1,
                        bestscore,
                        o_bestl1,
                        o_bestscore,
                        final_adv_images,
                    )

            self.adjust_constant(labels, bestscore, const, upper_bound, lower_bound)

        return final_adv_images


class AdaptiveJSMA(torchattacks.JSMA):
    def __init__(self, model, theta=1.0, gamma=0.1, eot_num=20):
        super().__init__(model, theta, gamma)
        self.eot_num = eot_num
        global grad_t_rdim
        grad_t_rdim = torch.empty(1)

    def compute_jacobian(self, image):
        var_image = image.clone().detach()
        var_image.requires_grad = True
        output = self.get_logits(var_image)
        num_features = int(np.prod(var_image.shape[1:]))
        jacobian = torch.zeros([output.shape[1], num_features], device=image.device)

        for _ in range(self.eot_num):
            output = self.get_logits(var_image)
            for i in range(output.shape[1]):
                if var_image.grad is not None:
                    var_image.grad.zero_()
                hook_handle_1 = self.model.gsem_module.t0.register_hook(hook_t0)
                hook_handle_2 = self.model.gsem_module.t_rdim.register_hook(hook_t_rdim)
                output[0][i].backward(retain_graph=True)
                hook_handle_1.remove()
                hook_handle_2.remove()
                # Copy the derivative to the target place
                jacobian[i] += (
                    var_image.grad.squeeze().view(-1, num_features).clone().squeeze()
                )  # nopep8

        jacobian /= self.eot_num

        return jacobian.to(self.device)
