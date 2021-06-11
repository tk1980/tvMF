import torch.nn as nn

from . import Loss

class WrapperNet(nn.Module):
    def __init__(self, model, num_classes, loss='CosLoss', **kwargs):
        super(WrapperNet, self).__init__()
        self.feat = model
        self.fc_loss = Loss.__dict__[loss](model.get_dim(), num_classes, **kwargs)
    
    def forward(self, x, target):
        x = self.feat(x)
        x, loss = self.fc_loss(x, target)
        return x, loss