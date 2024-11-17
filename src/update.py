import habana_frameworks.torch.core as htcore
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import DatasetSplit

class BenignUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.client_dataset = DatasetSplit(dataset, idxs)
        self.ldr_train = DataLoader(self.client_dataset, batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.unique_classes = set([int(label[0]) for label in self.client_dataset.dataset.labels])
        
    def restricted_softmax(self, logits):

        m_logits = torch.ones_like(logits).to(self.args.device) * self.args.fedrs_alpha
        class_mask = torch.tensor([c - 1 for c in self.unique_classes], dtype=torch.long).to(self.args.device)
        m_logits[:, class_mask] = 1.0
        logits = logits * m_logits

        return logits
    
    def train(self, net):
        global_net = copy.deepcopy(net)
        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)

        if self.args.method in ['fedsam', 'fedspeed']:
            minimizer = ESAM(net.parameters(), optimizer, rho=self.args.rho)
            
        for iter in range(self.args.local_ep):
            for batch, (images, labels) in enumerate(self.ldr_train):
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                logits = net(images)

                if self.args.method == 'fedrs':
                    logits = self.restricted_softmax(logits)
                
                if self.args.method in ['fedsam', 'fedspeed']:
                    minimizer.paras = [images, labels, self.loss_func, net]
                    minimizer.step()
                    if self.args.method == 'fedspeed':
                        loss_correct = 0
                        for pl, pg in zip(net.parameters(), global_net.parameters()):
                            loss_correct += torch.sum(torch.pow(pl - pg, 2))
                        loss_correct *= self.args.lamb
                        loss_correct.backward()
                        htcore.mark_step()
                else:
                    loss = self.loss_func(logits, labels.squeeze(dim=-1))
                
                # If using fedprox, apply the proximal term
                if self.args.method == 'fedprox':
                    prox_term = 0.0
                    for w, w_t in zip(net.parameters(), global_net.parameters()):
                        prox_term += ((w - w_t) ** 2).sum()
                    loss += 0.5 * self.args.lamb * prox_term
                
                if self.args.method not in ['fedsam', 'fedspeed']:
                    optimizer.zero_grad()
                    loss.backward()
                    htcore.mark_step()
                
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                htcore.mark_step()

        return net.state_dict()

class CompromisedUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        
    def train(self, net):

        net_freeze = copy.deepcopy(net)
        
        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
    
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                logits = net(images)
                
                loss = self.loss_func(logits, labels.squeeze(dim=-1))

                loss.backward()
                htcore.mark_step()
                
                optimizer.step()
                htcore.mark_step()

        for w, w_t in zip(net_freeze.parameters(), net.parameters()):
            w_t.data = (w_t.data - w.data) * self.args.mp_alpha
        
        return net.state_dict()

# ref: https://github.com/woodenchild95/FL-Simulator/tree/main
class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                p.requires_grad = True
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2)
                       if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, alpha=1.):
        inputs, labels, loss_func, net = self.paras

        if len(labels.shape) > 1:
            labels = labels.reshape(-1)

        predictions = net(inputs)

        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        htcore.mark_step()

        self.first_step()

        predictions = net(inputs)
        loss = alpha * loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        htcore.mark_step()

        self.second_step()
        htcore.mark_step()

    def _grad_norm(self):
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"]
              else 1.0) * p.grad).norm(p=2)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None]), p=2)
        return norm