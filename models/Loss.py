import torch
import torch.nn as nn
import torch.nn.functional as F


class CosLoss(nn.Linear):
    r"""
    Cosine Loss
    """
    def __init__(self, in_features, out_features, bias=False):
        super(CosLoss, self).__init__(in_features, out_features, bias)
        self.s_ = torch.nn.Parameter(torch.zeros(1))

    def loss(self, Z, target):
        s = F.softplus(self.s_).add(1.)
        l = F.cross_entropy(Z.mul(s), target, weight=None, ignore_index=-100, reduction='mean')
        return l
        
    def forward(self, input, target):
        logit = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), self.bias) # [N x out_features]
        l = self.loss(logit, target)
        return logit, l


class tvMFLoss(CosLoss):
    r"""
    t-vMF Loss
    """
    def __init__(self, in_features, out_features, bias=False, kappa=16):
        super(tvMFLoss, self).__init__(in_features, out_features, bias)
        self.register_buffer('kappa', torch.Tensor([kappa]))

    def forward(self, input, target=None):
        assert target is not None
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), None) # [N x out_features]
        logit =  (1. + cosine).div(1. + (1.-cosine).mul(self.kappa)) - 1.

        if self.bias is not None:
            logit.add_(self.bias)

        l = self.loss(logit, target)
        return logit, l
    
    def extra_repr(self):
        return super(tvMFLoss, self).extra_repr() + ', kappa={}'.format(self.kappa)