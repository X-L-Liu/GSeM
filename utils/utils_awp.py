import torch
from collections import OrderedDict
import torch.nn.functional as F

EPS = 1E-20

grad_t_rdim_awp = torch.empty(1)


def hook_t_rdim(grad):
    global grad_t_rdim_awp
    grad_t_rdim_awp = grad


def hook_t0(grad):
    global grad_t_rdim_awp
    return grad + grad_t_rdim_awp


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma, eot_num, use_amp):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.eot_num = eot_num
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        self.proxy_optim.zero_grad()
        for _ in range(self.eot_num):
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = - F.cross_entropy(self.proxy(inputs_adv), targets)
                loss = self.scaler.scale(loss)
            else:
                loss = - F.cross_entropy(self.proxy(inputs_adv), targets)
            hook_handle_1 = self.proxy.gsem_module.t0.register_hook(hook_t0)
            hook_handle_2 = self.proxy.gsem_module.t_rdim.register_hook(hook_t_rdim)
            loss.backward()
            hook_handle_1.remove()
            hook_handle_2.remove()
        if self.eot_num > 1:
            for p in self.proxy.parameters():
                p.grad.data /= self.eot_num
        if self.use_amp:
            self.scaler.step(self.proxy_optim)
            self.scaler.update()
        else:
            self.proxy_optim.step()
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)
