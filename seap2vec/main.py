# -*- coding: utf-8 -*-
"""

not CLI package

@author: tadahaya
"""
# packages installed in the current environment
import os
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import glob

# original packages in src
from .src import utils
from .src import data_handler as dh
from .src.models import VAE
from .src.models import loss_function

SEP = os.sep
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SEAP2vec:
    def __init__(
            self, workdir:str="", datafile:str="", dim_img:int=1,
            seed:int=222, num_epoch:int=100, batch_size:int=128, lr:float=1e-4,
            n_monitor:int=1000, dim_latent=64, beta=0.1
            ):
        self.workdir = workdir
        self.dim_img = dim_img
        if self.dim_img==1:
            self.pooling_kernels = (2, 2)
            self.enc_out = 16
        elif self.dim_img==2:
            self.pooling_kernels = (4, 4)
            self.enc_out = 16
        else:
            raise ValueError("!! dim_img should be 1 or 2 !!")
        self.seed = 222
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.dim_latent = dim_latent
        self.beta = beta
        self.n_monitor = n_monitor
        utils.fix_seed(seed=seed, fix_gpu=False) # for seed control
        self._now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.dir_name = self.workdir + SEP + 'results' + SEP + self._now # for output
        self.datafile = datafile
        if not os.path.exists(self.datafile):
            tmp = sorted(glob.glob(self.workdir + SEP + '*.npz'), reverse=True)[0]
            if len(tmp) > 0:
                self.datafile = tmp
                print(f"load the following data: {tmp}")
            else:
                raise ValueError("!! Give data path !!")
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        self.logger = utils.init_logger(
            __name__, self.dir_name, self._now, level_console='debug', level_file='debug'
            )
        self.model = None


    def prepare_data(self):
        """
        データの読み込み・ローダーの準備を実施
        加工済みのものをdataにおいておくか, argumentで指定したパスから呼び出すなりしてデータを読み込む
        inference用を読み込む際のものも用意しておくと楽
        
        """
        dataset = np.load(self.datafile)
        input = dataset["input"]
        output = dataset["output"]
        idx = int(input.shape[0] * 0.9)
        input = np.transpose(input, [0,3,1,2]) # nhwc -> nchw
        output = np.transpose(output, [0,3,1,2]) # nhwc -> nchw
        input = torch.tensor(input).float()
        output = torch.tensor(output).float()
        if self.dim_img == 1:
            tfn_train = transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.0), fill=255),
                transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0))
                ])
            tfn_test = transforms.Compose([
                transforms.GaussianBlur(kernel_size=3, sigma=1.0)
                ])
            # 1Dの場合はx軸方向のみ不変
        elif self.dim_img == 2:
            tfn_train = transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255),
                transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0))
                ])
            tfn_test = transforms.Compose([
                transforms.GaussianBlur(kernel_size=3, sigma=1.0)
                ])
            # 2Dの場合はx,y両方不変
        train_loader, test_loader = dh.prep_data(
            input[:idx], output[:idx], input[idx:], output[idx:],
            batch_size=self.batch_size, transform=(tfn_train, tfn_test)
            )
        return train_loader, test_loader


    def prepare_model(self):
        """
        model, loss, optimizer, schedulerの準備

        """
        model = VAE(
            color_channels=1, pooling_kernels=self.pooling_kernels,
            encoder_output_size=self.enc_out, dim_latent=self.dim_latent
            )
        model.to(DEVICE)
        criterion = loss_function
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epoch, eta_min=0
            )
        return model, criterion, optimizer, scheduler


    def train_epoch(self, model, train_loader, test_loader, criterion, optimizer):
        """
        epoch単位の学習構成, なくとも良い
        
        """
        model.train() # training
        train_batch_loss = []
        train_batch_rl = []
        train_batch_kld = []
        for data_in, data_out in train_loader:
            data_in, data_out = data_in.to(DEVICE), data_out.to(DEVICE) # put data on GPU
            optimizer.zero_grad() # reset gradients
            output, mu, logvar = model(data_in) # forward
            loss, rl, kld = criterion(output, data_out/255, mu, logvar, self.beta) # calculate loss
            # data_outが[0, 255]なので, BCElossを使うためには255で割り込む必要がある
            loss.backward() # backpropagation
            optimizer.step() # update parameters
            train_batch_loss.append(loss.item())
            train_batch_rl.append(rl.item())
            train_batch_kld.append(kld.item())    
        model.eval() # test (validation)
        test_batch_loss = []
        test_batch_rl = []
        test_batch_kld = []
        with torch.no_grad():
            for data_in, data_out in test_loader:
                data_in, data_out = data_in.to(DEVICE), data_out.to(DEVICE)
                output, mu, logvar = model(data_in)
                loss, rl, kld = criterion(output, data_out/255, mu, logvar, self.beta)
                test_batch_loss.append(loss.item())
                test_batch_rl.append(rl.item())
                test_batch_kld.append(kld.item())
        train_loss = (
            np.mean(train_batch_loss) / self.batch_size,
            np.mean(train_batch_rl) / self.batch_size,
            np.mean(train_batch_kld) / self.batch_size
            )
        test_loss = (
            np.mean(test_batch_loss) / self.batch_size,
            np.mean(test_batch_rl) / self.batch_size,
            np.mean(test_batch_kld) / self.batch_size
            )
        return model, train_loss, test_loss


    def fit(self, model, train_loader, test_loader, criterion, optimizer, scheduler):
        """
        学習
        model, train_loss, test_loss (valid_loss)を返す
        
        """
        train_loss = []
        train_rl = []
        train_kld = []
        test_loss = []
        test_rl = []
        test_kld = []
        for epoch in range(self.num_epoch):
            model, train_epoch_loss, test_epoch_loss = self.train_epoch(
                model, train_loader, test_loader, criterion, optimizer
                )
            scheduler.step() # should be removed if not necessary
            train_loss.append(train_epoch_loss[0])
            train_rl.append(train_epoch_loss[1])
            train_kld.append(train_epoch_loss[2])
            test_loss.append(test_epoch_loss[0])
            test_rl.append(test_epoch_loss[1])
            test_kld.append(test_epoch_loss[2])
            if epoch % self.n_monitor == 0:
                self.logger.info(
                    f'epoch: {epoch} // train_loss: {train_epoch_loss[0]:.4f} // valid_loss: {test_epoch_loss[0]:.4f}'
                    )
                print(
                    f'epoch: {epoch} // train_loss: {train_epoch_loss[0]:.4f} // valid_loss: {test_epoch_loss[0]:.4f}'
                )
        return model, (train_loss, train_rl, train_kld), (test_loss, test_rl, test_kld)


    def main(self):
        start = time.time() # for time stamp
        # 1. data prep
        train_loader, test_loader = self.prepare_data()
        self.logger.info(
            f'num_training_data: {len(train_loader)}, num_test_data: {len(test_loader)}'
            )
        # 2. model prep
        model, criterion, optimizer, scheduler = self.prepare_model()
        # 3. training
        self.model, train_loss, test_loss = self.fit(
            model, train_loader, test_loader, criterion, optimizer, scheduler
            )
        utils.plot_progress(
            train_loss[0], test_loss[0], self.num_epoch, self.dir_name, xlabel="epoch", ylabel="loss"
            )
        utils.plot_progress(
            train_loss[1], test_loss[1], self.num_epoch, self.dir_name, xlabel="epoch", ylabel="RL"
            )
        utils.plot_progress(
            train_loss[2], test_loss[2], self.num_epoch, self.dir_name, xlabel="epoch", ylabel="KLD"
            )
        utils.summarize_model(model, next(iter(train_loader))[0], self.dir_name)
        utils.save_loss(train_loss[0], test_loss[0], self.dir_name, "total")
        utils.save_loss(train_loss[1], test_loss[1], self.dir_name, "RL")
        utils.save_loss(train_loss[2], test_loss[2], self.dir_name, "KLD")
        # 4. save results & config
        utils.to_logger(self.logger, name='loss', obj=criterion)
        utils.to_logger(
            self.logger, name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
            )
        self.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))


    def load_model(self, url):
        self.model, _, _, _ = self.prepare_model()
        self.model.load_state_dict(torch.load(url))


    def inference(self, x):
        # load
        if self.model is None:
            raise ValueError("!! No trained model !!")
        input = np.transpose(x, [0,3,1,2])
        input = torch.tensor(input).float()
        # prep dataloader
        dataset = dh.prep_dataset(input, input, None)
        dataloader = dh.prep_dataloader(dataset, self.batch_size, shuffle=False)
        self.model.eval() # test (validation)
        mus = []
        logvars = []
        outputs = []
        with torch.no_grad():
            for data_in, data_out in dataloader:
                data_in, data_out = data_in.to(DEVICE), data_out.to(DEVICE)
                output, mu, logvar = self.model(data_in)
                mus.append(mu)
                logvars.append(logvar)
                outputs.append(output)
        mus = torch.cat(mus, axis=0)
        logvars = torch.cat(logvars, axis=0)
        outputs = torch.cat(outputs, axis=0)
        if torch.cuda.is_available():
            mus = mus.cpu().detach().numpy()
            logvars = logvars.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
        return mus, logvars, outputs
    

    def imshow(self, data, cmap='binary_r', figsize=None):
        """ show pixelized data """
        plt.figure(figsize=figsize)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        plt.imshow(data, aspect='equal', cmap=cmap)
        plt.show()