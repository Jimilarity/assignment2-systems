from __future__ import annotations

import math

import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {'lr': lr})
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                p.data -= -lr / math.sqrt(t + 1) * p.grad.data
                state['t'] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if not state:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                state['t'] += 1
                t = state['t']
                m, v = state['m'], state['v']
                # 因为梯度更新调用次数多，因此对性能要求极高，使用add_() ,addcmul_(), addcdiv_()等函数，
                # 少创建中间张量，省显存，同时单次 kernel 调用，更快
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data.mul_(1 - lr * weight_decay)
                p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
        return loss
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warm_up_iters: int,
    consine_cycle_iters: int,
):
    if it < warm_up_iters:
        return it / warm_up_iters * max_learning_rate
    if it > consine_cycle_iters:
        return min_learning_rate
    progress = (it - warm_up_iters) / (consine_cycle_iters - warm_up_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)