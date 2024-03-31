# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer

ToDo BT用のtrainerを作成する

@author: tadahaya
"""
import torch
from torch import nn, optim

from .utils import save_experiment, save_checkpoint
from .data_handler import prepare_data # ToDo BT用に変更する

class Trainer:
    def __init__(self, config, model, optimizer, loss_fn, exp_name, device):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device


    def train(self, trainloader, testloader, save_model_evry_n_epochs=0):
        """
        train the model for the specified number of epochs.
        
        """
        # configの確認
        # ToDo BT用に内容を変更する
        config = self.config
        assert config["hidden_size"] % config["num_attention_heads"] == 0
        assert config["intermediate_size"] == 4 * config["hidden_size"]
        assert config["image_size"] % config["patch_size"] == 0
        # keep track of the losses
        train_losses, test_losses = [], []
        # training
        for i in range(config["epochs"]):
            train_loss = self.train_epoch(trainloader)
            test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}"
                )
            if save_model_evry_n_epochs > 0 and (i + 1) % save_model_evry_n_epochs == 0 and i + 1 != config["epochs"]:
                print("> Save checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        # save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, [])
        # accuracy消した分の整合性


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        for (x1, x2), _ in trainloader: # label不要
            # batchをdeviceへ
            x1, x2 = x1.to(self.device), x2.to(self.device)
            # 勾配を初期化
            self.optimizer.zero_grad()
            # forward
            output = self.model(x1, x2) # BarlowTwinsのforward
            # loss計算
            loss = self.loss_fn(output) # label不要
            # backpropagation
            loss.backward()
            # パラメータ更新
            self.optimizer.step()
            total_loss += loss.item() * len(x1) # loss_fnがbatch内での平均の値になっている模様
        return total_loss / len(trainloader.dataset) # 全データセットのうちのいくらかという比率になっている
    

    @torch.no_grad()
    def evaluate(self, testloader):
        """
        あくまでSSLのtest, lossの推移だけ見る
        
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for (x1, x2), _ in testloader:
                # batchをdeviceへ
                x1, x2 = x1.to(self.device), x2.to(self.device)
                # 予測
                output = self.model(x1, x2)
                # lossの計算
                loss = self.loss_fn(output)
                total_loss += loss.item() * len(x1)
        avg_loss = total_loss / len(testloader.dataset)
        return avg_loss
    

class ClassifierTrainer:
    """ BTを使ってclassifierをlinear taskするモデルのtrainer """
    def __init__(self, config, model, optimizer, loss_fn, exp_name, device):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device


    def train(self, trainloader, testloader, save_model_evry_n_epochs=0):
        """
        train the model for the specified number of epochs.
        
        """
        # configの確認
        # ToDo BT用に内容を変更する
        config = self.config
        assert config["hidden_size"] % config["num_attention_heads"] == 0
        assert config["intermediate_size"] == 4 * config["hidden_size"]
        assert config["image_size"] % config["patch_size"] == 0
        # keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # training
        for i in range(config["epochs"]):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
                )
            if save_model_evry_n_epochs > 0 and (i + 1) % save_model_evry_n_epochs == 0 and i + 1 != config["epochs"]:
                print("> Save checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        # save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        for (x1, x2), _ in trainloader: # label不要
            # batchをdeviceへ
            x1, x2 = x1.to(self.device), x2.to(self.device)
            # 勾配を初期化
            self.optimizer.zero_grad()
            # forward
            output = self.model(x1, x2) # BarlowTwinsのforward
            # loss計算
            loss = self.loss_fn(output) # label不要
            # backpropagation
            loss.backward()
            # パラメータ更新
            self.optimizer.step()
            total_loss += loss.item() * len(x1) # loss_fnがbatch内での平均の値になっている模様
        return total_loss / len(trainloader.dataset) # 全データセットのうちのいくらかという比率になっている
    

    @torch.no_grad()
    def evaluate(self, testloader):
        """
        あくまでSSLのtest, lossの推移だけ見る
        
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for (x1, x2), _ in testloader:
                # batchをdeviceへ
                x1, x2 = x1.to(self.device), x2.to(self.device)
                # 予測
                output = self.model(x1, x2)
                # lossの計算
                loss = self.loss_fn(output)
                total_loss += loss.item() * len(x1)
        avg_loss = total_loss / len(testloader.dataset)
        return avg_loss