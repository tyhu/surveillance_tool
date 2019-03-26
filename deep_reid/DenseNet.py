from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['DenseNet121', 'DenseNetDist121']

class AttentionPooling(nn.Module):
    def __init__(self, feat_dim=1024):
        super(AttentionPooling, self).__init__()
        self.layer = nn.Linear(feat_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.feat_dim = feat_dim

    ### feats: size(batch_size, h, w, feat_dim)
    def forward(self, feats):
        b,_,h,w = feats.size()
        feats = feats.permute(0,2,3,1).view(b,h*w,-1)
        a = self.layer(feats)
        a = self.softmax(a)
        a = a.expand(b, h*w, self.feat_dim)
        att_feats = torch.mul(feats,a)
        att_feats = torch.sum(att_feats,1)
        return att_feats

class DenseNet121(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(DenseNet121, self).__init__()
        self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)
        self.feat_dim = 1024 # feature dimension
        self.att = AttentionPooling(feat_dim=self.feat_dim)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        #x = F.avg_pool2d(x, (2,2))
        #x = self.att(x)
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)
        
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def GaussianPooling(x):
    m = torch.mean(x, dim=1)
    s = torch.std(x, dim=1)
    return m, s

class DenseNetDist121(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(DenseNetDist121, self).__init__()
        self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)
        self.feat_dim = 1024 # feature dimension
        self.att = AttentionPooling(feat_dim=self.feat_dim)

    def forward(self, x):
        x = self.base(x)

        x2 = F.avg_pool2d(x, (1,2))
        #x2 = x
        b, _, h, w = x2.size()
        x2 = x2.permute(0,2,3,1).view(b, h*w, -1)
        m, s = GaussianPooling(x2)

        x = F.avg_pool2d(x, x.size()[2:])
        #x = self.att(x)
        f = x.view(x.size(0), -1)
        if not self.training:
            return m, s
        y = self.classifier(f)
        
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
