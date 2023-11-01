# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

inspired by the following:
https://gist.github.com/koshian2/64e92842bec58749826637e3860f11fa

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            kernel, stride, padding
            ):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel, stride=stride, padding=padding
            )
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        h = self.conv(x)
        return torch.relu(self.bn(h))


class DecoderBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            stride, activation:str="relu"
            ):
        super(DecoderBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=stride, stride=stride
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "relu":
            self.a = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.a = nn.Sigmoid() # ここはinplaceないか
        else:
            raise KeyError("!! activation should be relu or sigmoid!!")


    def forward(self, x):
        h = self.convt(x)
        return self.a(self.bn(h))


class Encoder(nn.Module):
    def __init__(self, color_channles, pooling_kernels):
        super(Encoder, self).__init__()
        ## botlle neck部分のサイズと途中のkernel sizeとの関係で決まる
        self.b1 = EncoderBlock(color_channles, 32, kernel=1, stride=1, padding=0)
        ## 基本バイナリ画像のためin_channels=1
        self.b2 = EncoderBlock(32, 64, kernel=3, stride=1, padding=0)
        self.b3 = EncoderBlock(64, 128, kernel=3, stride=pooling_kernels[0], padding=1)
        ## kernel検討の余地あり, 2
        self.b4 = EncoderBlock(128, 256, kernel=3, stride=pooling_kernels[1], padding=1)
        ## kernel検討の余地あり, 2


    def forward(self, x):
        h = self.b1(x)
        h = self.b2(h)
        h = self.b3(h)
        return self.b4(h)


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels):
        super(Decoder, self).__init__()
        self.b1 = DecoderBlock(256, 128, stride=1)
        self.b2 = DecoderBlock(128, 64, stride=pooling_kernels[1]) # corresponds to Encoder.b3
        self.b3 = DecoderBlock(64, 32, stride=pooling_kernels[0]) # corresponds to Encoder.b3
        self.b4 = DecoderBlock(32, color_channels, stride=1, activation="sigmoid")


    def forward(self, x):
        h = self.b1(x)
        h = self.b2(h)
        h = self.b3(h)
        return self.b4(h)


class VAE(nn.Module):
    def __init__(
            self, color_channels, pooling_kernels,
            encoder_output_size, dim_latent
            ):
        super(VAE, self).__init__()
        self.encoder_output_size = encoder_output_size
        self.n_middle_neurons = 256 * encoder_output_size * encoder_output_size
        ## inputと途中のkernel sizeで決まる
        self.dim_latent = dim_latent
        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernels)
        # Middle layers
        self.fc_mu = nn.Linear(
            self.n_middle_neurons, dim_latent
            )
        self.fc_logvar = nn.Linear(
            self.n_middle_neurons, dim_latent
            )
        self.fc_to_decoder = nn.Linear(
            dim_latent, self.n_middle_neurons
        )        
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernels)


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.n_middle_neurons)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    def decode(self, z):
        h = self.fc_to_decoder(z)
        h = h.view(-1, 256, self.encoder_output_size, self.encoder_output_size)
        return self.decoder(h)


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # RL = F.binary_cross_entropy(recon_x, x, size_average=False)        
    criterion = nn.BCELoss(size_average=False)
    RL = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RL + KLD, RL, KLD