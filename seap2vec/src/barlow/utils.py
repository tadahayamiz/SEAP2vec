# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from .models import VitForClassification

def save_experiment(
        experiment_name, config, model, train_losses, test_losses,
        accuracies, base_dir=""
        ):
    if len(base_dir) == 0:
        base_dir = os.path.dirname(config["config_path"])
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    # save config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # save metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)


def load_experiments(
        experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"
        ):
    outdir = os.path.join(base_dir, experiment_name)
    # load config
    configfile = os.path.join(outdir, "config.json")
    with open(configfile, 'r') as f:
        config = json.load(f)
    # load metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data["train_losses"]
    test_losses = data["test_losses"]
    accuracies = data["accuracies"]
    # load model
    model = VitForClassification(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile)) # checkpointを読み込んでから
    return config, model, train_losses, test_losses, accuracies


def visualize_images(nrow:int=5, ncol:int=6):
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True
    )
    classes = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )
    # randomに選択
    indices = torch.randperm(len(trainset))[:nrow * ncol]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # 描画
    fig = plt.figure()
    for i in range(nrow * ncol):
        ax = fig.add_subplot(ncol, nrow, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])
    

@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    """
    visualize the attention of the first 4 images
    
    """
    model.eval()
    # randomに選択
    num_images = 30
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    classes = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )
    indices = torch.randperm(len(testset))[:30]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    # image -> tensor
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    images = torch.stack([test_transform(image) for image in raw_images])
    # imageをdeviceに載せる
    images = images.to(device)
    model = model.to(device)
    # 全ブロックのattention mapを最終ブロックから取得 (appendされてる)
    logits, attention_maps = model(images, output_attentions=True)
    ## att_maps = [(batch, head, token, token), ...]
    # predictionを取得
    predictions = torch.argmax(logits, dim=1)
    # attention blockをheadの軸でconcatする
    attention_maps = torch.cat(attention_maps, dim=1)
    # CLS tokenのものだけ抽出
    attention_maps = attention_maps[:, :, 0, 1:]
    # -> (batch, block, token - 1) = (batch, block, patch)
    ## cls tokenは先頭なので先頭以外をとってきている
    # 全blockについてCLStokenのattention mapsの平均をとる
    attention_maps = attention_maps.mean(dim=1)
    # -> (batch, patch)
    # attention mapをsquareへ変換
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # attention mapを元の画像サイズに戻す
    attention_maps = attention_maps.unsqueze(1) # channelをunsqueezeしてから戻す
    attention_maps = F.interpolate(
        attention_maps, size=(32, 32), mode="bilinear", align_corners=False
        )
    attention_maps = attention_maps.squeeze(1)
    # 描画
    fig = plt.figure(figsize=(20, 10))
    # 2つのimageを用意
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # 左側のimageについてattention mapをmask
        extended_attention_map = np.concatenate(
            (np.zeros((32, 32)), attention_maps[i].cpu()), axis=1
            )
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # ground truthとpredictedを載せる
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()