# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 2024

Barlow Twins model
inspired from https://github.com/MaxLikesMath/Barlow-Twins-Pytorch

@author: tadahaya
"""

import torch
import torch.nn as nn

def flatten(t):
    return t.reshape(t.shape[0], -1)

class NetWrapper(nn.Module):
    """ inspired from https://github.com/lucidrains/byol-pytorch """
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer)==str: # 名称で取得
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer)==int: # indexで取得
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output): # hookでflattenする
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"!! Hidden layer ({self.layer}) not found !!"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if self.layer==-1:
            return self.net(x)
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None # self.hiddenを初期化している
        assert hidden is not None, f"!! Hidden layer ({self.layer}) never emitted an output !!"
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        return representation

def off_diagonal(x):
    """ return a flattened view of the off-diagonal elements of a square matrix """
    n, m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    """
    single GPU version based on https://github.com/facebookresearch/barlowtwins

    """
    def __init__(self, backbone, latent_id, projection_sizes, lambd, scale_factor=1):
        """
        Parameters
        ----------
        backbone: Model

        latent_id: name or index of the layer to be fed to the projection

        projection_sizes: size of the hidden layers in the projection

        lambd: tradeoff function

        scale_factor: factor to scale loss by

        """
        super().__init__()
        self.backbone = backbone
        self.backbone = NetWrapper(self.backbone, latent_id)
        self.lambd = lambd
        self.scale_factor = scale_factor
        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False)) # BatchNorm入れるのでbias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False)) # BatchNorm入れるのでbias=False
        self.projector = nn.Sequential(*layers)
        # normalization layer for z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])
        # scaling
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss