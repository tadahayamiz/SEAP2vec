# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

prepare dataset

@author: tadahaya
"""
import time
import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# import cv2
from collections import Counter
from itertools import chain
from tqdm.auto import tqdm, trange

FIELD = [
    "data_id",	"well_id", "value", "date", "color", "plate",
    "sample_id", "sample_name", "sample_label", "slice", "X", "Y"
    ]

SEP = os.sep

class Preprocess:
    """
    preprocessor, 基本的にハード
    全部こみこみのtableデータからメタ情報を抜く
    
    基本的にsample_idが1測定になっている
    ただしsample_nameと1 : 1にはなっていない
    sample_labelが疾患状態を表す

    Parameters
    ----------
    url: str
        filepath

    colors: str
        前処理が必要な色
    
    beta: float
        上側外れ値のthrehold
    
    """
    def __init__(
            self, url, colors=["green", "red"], beta:float=2.0, fileout:str="", sep:str=","
            ):
        df = pd.read_csv(url, sep=sep)
        # meta information
        focused = ["sample_id", "sample_name", "sample_label"]
        df2 = df.copy().loc[:, focused].drop_duplicates()
        names = list(df2["sample_name"])
        ids = list(df2["sample_id"])
        new = [f"{n}_{i}" for n, i in zip(names, ids)]
        df2.loc[:, "unique_name"] = new
        self.meta = df2
        # preprocessing
        # self._fix(v_name)
        # data preprocessing
        self.beta = beta
        df3 = df.copy()
        ## 下側外れ値
        df3 = df3[df3["value"] > 0]
        conv = []
        for i in ids:
            tmp0 = df3[df3["sample_id"]==i]
            for c in colors:
                tmp = tmp0[tmp0["color"]==c]
                x = tmp["value"].values.flatten()
                q3, q1 = np.percentile(x, (75, 25))
                iqr = q3 - q1
                tmp = tmp[tmp["value"] <= q3 + beta * iqr]
                conv.append(tmp)
        df3 = pd.concat(conv, axis=0, join="inner")
        print(f"complete preprocessing: {df.shape} -> {df3.shape}")
        self.data = df3
        # export
        if len(fileout) == 0:
            ext = url.split(".")[-1]
            fileout0 = url.replace(f".{ext}", f"_meta.{ext}")
            fileout1 = url.replace(f".{ext}", f"_data_{int(beta)}IQR.{ext}")
        self.meta.to_csv(fileout0, sep=sep)
        self.data.to_csv(fileout1, sep=sep)


    def _fix(self, v_name:str="FITC"):
        """ mainの値であるFITCに変なstrが混じっていたので対処 """
        before = self.data.shape
        def convert(x):
            try:
                return float(x)
            except ValueError:
                return np.nan
        self.data.loc[:, v_name] = self.data.loc[:, v_name].map(convert)
        self.data = self.data.dropna(subset=[v_name])
        # print(before, self.data.shape)


    def check_content(self):
        """ 中身を確認する """
        count = []
        for c in self.id_col:
            print("====================================", c)
            tmp = self.data[c].values.flatten().tolist()
            cntr = Counter(tmp)
            count.append(cntr)
            for k, v in cntr.items():
                print(k, v)


class Data:
    """
    data格納モジュール, 基本的にハード
        
    """
    def __init__(self, input, colors:list=["green", "red"]):
        # 読み込み
        self.data = None
        self.colors = colors
        self.dim = len(colors)
        assert (self.dim > 0) & (self.dim <= 2)
        if type(input) == str:
            self.data = pd.read_csv(input, index_col=0)
        elif type(input) == type(pd.DataFrame()):
            self.data = input
        else:
            raise ValueError("!! Provide url or dataframe !!")            
        # データの中身の把握
        col = list(self.data.columns)
        self.dic_components = dict()
        for c in col:
            tmp = self.data[c].values.flatten().tolist()
            self.dic_components[c] = Counter(tmp)


    def conditioned(self, condition:dict):
        """ 解析対象とするデータを条件付けする """
        for k, v in condition.items():
            try:
                self.data = self.data[self.data[k]==v]
            except KeyError:
                raise KeyError("!! Wrong key in condition: check the keys of condition !!")


    def sample1(
        self, sid:int, n_sample:int=256, ratio:float=0.9,
        v_name:str="value", s_name:str="sample_id"
        ):
        """
        指定した検体からn_sampleの回の輝点のサンプリングを行う

        Parameters
        ----------
        sid: int
            Specimen ID, 検体をidentifyする

        v_name: str
            valueカラムの名称
            DPPVIの場合は素直にvalue

        Returns
        -------
        a list of sampled data

        """
        tmp = self.data[self.data[s_name]==sid]
        tmp = tmp[tmp["color"]==self.colors[0]]
        n = int(tmp.shape[0] * ratio)        
        res = [tmp.sample(n=n)[v_name].values for i in range(n_sample)]
        return res


    def sample2(
        self, sid:int, n_sample:int=256, ratio:float=0.9,
        v_name:str="value", s_name:str="sample_id"
        ):
        """
        指定した検体からn_sampleの回の輝点のサンプリングを行う

        Parameters
        ----------
        sid: int
            Specimen ID, 検体をidentifyする

        v_name: str
            valueカラムの名称
            DPPVIの場合は素直にvalue

        Returns
        -------
        a list of sampled data

        """
        tmp = self.data[self.data[s_name]==sid]
        tmp0 = tmp[tmp["color"]==self.colors[0]]
        tmp1 = tmp[tmp["color"]==self.colors[1]]
        common_well = set(tmp0["well_id"]) & set(tmp1["well_id"])
        tmp0 = tmp0[tmp0["well_id"].isin(common_well)]
        tmp1 = tmp1[tmp1["well_id"].isin(common_well)] # sample_idとcolorを絞るとwell_idのサイズに一致
        x0 = tmp0[v_name].values
        x1 = tmp1[v_name].values
        X = np.array([x0, x1]).T
        n = int(tmp0.shape[0] * ratio) 
        idx = list(range(n))
        rng = np.random.default_rng()
        res = []
        for i in range(n_sample):
            tmp_idx = idx.copy()
            rng.shuffle(tmp_idx)
            res.append(X[tmp_idx, :])
        return res


    def prep_data(
            self, samplesize:int=10000, ratio:float=0.9,
            shuffle:bool=True, v_name:str="value", s_name:str="sample_id"
            ):
        """
        指定したsamplesizeまでサンプリングを行う
        dataをlistで返す
        1dなら[1d-array]
        2dなら[2d-array]
        
        """
        specimens = set(list(self.data[s_name]))
        print(f"> handling {len(specimens)} specimens")
        n_sample = samplesize // len(specimens) # n_sampleを決める
        res = []
        specimen = []
        if self.dim == 1:
            for s in tqdm(specimens):
                tmp = self.sample1(s, n_sample, ratio, v_name, s_name)
                res.append(tmp)
                specimen.append([s] * n_sample)
        elif self.dim == 2:
            for s in tqdm(specimens):
                tmp = self.sample2(s, n_sample, ratio, v_name, s_name)
                res.append(tmp)
                specimen.append([s] * n_sample)
        else:
            raise ValueError("!! check |colors|, which should be 1 or 2 !!")
        res = list(chain.from_iterable(res))
        specimen = list(chain.from_iterable(specimen))
        if shuffle:
            rng = np.random.default_rng()
            idx = list(range(len(res)))
            rng.shuffle(idx)
            res = [res[i] for i in idx]
            specimen = [specimen[i] for i in idx]
        
        print(len(res), len(specimen))

        return res, specimen
    

    def imshow1(
            self, sid:int, bins:int=64, ratio:float=0.9, condition:dict=dict(),
            outdir:str="", v_name:str="value", s_name:str="sample_id",
            figsize=(), fontsize:int=16
            ):
        """ 指定したIDの画像を表示する """
        # data
        if len(condition) > 0:
            self.conditioned(condition)
        data = self.sample1(sid, 2, ratio, v_name, s_name)[0]
        # show
        if len(figsize) > 0:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        plt.rcParams["font.size"] = fontsize
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        ax.hist(data, color="black", bins=bins)
        if len(outdir) > 0:
            plt.savefig(outdir + SEP + f"hist_{sid}.png")
        plt.show()


    def imshow2(
            self, sid:int, symbol_size:int=30, symbol_alpha:float=0.5,
            linewidths:float=0.0, pixel:tuple=(64, 64), dpi:int=100,
            ratio:float=0.9, condition:dict=dict(),
            outdir:str="", v_name:str="value", s_name:str="sample_id",
            figsize=(), fontsize:int=16
            ):
        """ 指定したIDの画像を表示する """
        # data
        if len(condition) > 0:
            self.conditioned(condition)
        data = self.sample2(sid, 2, ratio, v_name, s_name)[0]
        # show
        if len(figsize)==0:
            figsize = pixel[0] / dpi, pixel[1] / dpi
        fig = plt.figure(figsize=figsize)
        plt.rcParams["font.size"] = fontsize
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        ax.scatter(
            data[sid][:, 0], data[sid][:, 1],
            color="black", s=symbol_size, alpha=symbol_alpha,
            linewidths=linewidths
            )
        if len(outdir) > 0:
            plt.savefig(outdir + SEP + f"scatter_{sid}.png")
        plt.show()


class DataMaker:
    def __init__(self):
        self.pixel = None
        self._dpi = None
        self.data = None
        self.specimen = None

    def set_data(self):
        """ abstract method """
        raise NotImplementedError
    
    def set_specimen(self, specimen):
        """ specimen ID """
        self.specimen = specimen

    def main(self):
        """ abstract method """
        raise NotImplementedError        

    def imshow(self):
        """ abstract method """
        raise NotImplementedError        

      
class HistMaker(DataMaker):
    """
    dataを読み込んでヒストグラムを表すnp配列へと変換する
    
    """
    def __init__(self, pixel:tuple=(64, 64), bins:tuple=(64, 32)):
        super().__init__()
        self.pixel = pixel
        self._dpi = 100
        self._figsize = pixel[0] / self._dpi, pixel[1] / self._dpi
        self.bins = bins
        self.data = None
        self.specimen = None
        self.limit = None


    def set_data(self, data, test_bins=None, limit:tuple=()):
        """ setter """
        self.data = data
        if len(limit) == 0:
            vmin = [np.min(v) for v in data]
            vmax = [np.max(v) for v in data]
            limit = (np.min(vmin), np.max(vmax))
        self.limit = limit
        # plot
        if test_bins is None:
            test_bins = self.bins[0]
        self._test_view(data, test_bins, limit)

    
    def main(
            self, outdir:str="", name:str="", ratio:float=0.9,
            bins:tuple=(), test_view:bool=True, limit:tuple=()
            ):
        """
        dataをまとめてhistogram arrayへと変換, npzで保存する
        input array, output arrayの順
        各arrayはsample, h, wの順
        time consuming, 1000回すのに5 min程度かかる
        
        Parameters
        ----------
        ratio: float, (0, 1)
            サンプルからデータ点を取得する割合
        
        bins: tuple
            前者がinput, 後者がoutputのヒストグラムのbinsを指定する
            AE的に組むため, 前者が大きいことが前提

        """
        assert len(outdir) > 0
        assert (ratio > 0) & (ratio < 1) 
        # dataの準備
        array0 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        array1 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        if len(bins) == 0:
            bins = self.bins
        assert bins[0] >= bins[1]
        for i, d in tqdm(enumerate(self.data)):
            # dataのhistogram化
            img0 = self.get_hist_array(d, bins=bins[0])
            img1 = self.get_hist_array(d, bins=bins[1])
            # imageのbinarize化と格納
            array0[i, :, :] = self.binarize(img0)
            array1[i, :, :] = self.binarize(img1)
        # NHWC形式になるように次元を追加
        array0 = array0[..., np.newaxis]
        array1 = array1[..., np.newaxis]
        # npzで保存
        if test_view:
            print("input example")
            self.imshow(array0[0, :, :, 0])
            print("output example")
            self.imshow(array1[0, :, :, 0])
        now = datetime.datetime.now().strftime('%Y%m%d')
        if len(name) > 0:
            fileout = outdir + SEP + f"dataset_{name}_{now}.npz"
        else:
            fileout = outdir + SEP + f"dataset_{now}.npz"
        np.savez_compressed(
            fileout, input=array0, output=array1, specimen=self.specimen
            )


    def get_hist_array(self, data, bins=30):
        """
        データをヒストグラムへと変換し, そのarrayを得る
        
        """
        # prepare histogram
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        # ax.hist(
        #     data, color="black", density=True, bins=bins, range=self.limit
        #     )
        # rangeを考慮しない方針
        ax.hist(data, color="black", bins=bins)
        # convert array
        fig.canvas.draw() # レンダリング
        data = fig.canvas.tostring_rgb() # rgbのstringとなっている
        w, h = fig.canvas.get_width_height()
        c = len(data) // (w * h) # channelを算出
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
        plt.close()
        return img


    def binarize(self, data):
        """ 得られたarrayを二値化する """
        data = data.sum(axis=2) # h, w, cであり, blackなので255, 0のみとなっている
        data = np.where(data > 0, 255, 0)
        return data


    def imshow(self, data, cmap='binary_r', figsize=None):
        """ show pixelized data """
        plt.figure(figsize=figsize)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        plt.imshow(data, aspect='equal', cmap=cmap)
        plt.show()

    
    def _test_view(self, data, test_bins, limit):
        """ refer to set_data """
        num = 4
        idx = list(range(len(data)))
        np.random.shuffle(idx)
        fig, axes = plt.subplots(1, num, figsize=(2.5 * num, 2.5))
        for i in range(num):
            # axes[i].hist(
            #     data[idx[i]], color="black", density=True,
            #     bins=test_bins, range=limit
            #     )
            # rangeを考慮しない方針 230918
            axes[i].hist(
                data[idx[i]], color="black", bins=test_bins
                )

        plt.show()


class ScatterMaker(DataMaker):
    """
    dataを読み込んで散布図を表すnp配列へと変換する
    
    """
    def __init__(
            self, pixel:tuple=(64, 64), symbol_size:tuple=(32, 64),
            symbol_alpha:tuple=(0.5, 0.3)
            ):
        super().__init__()
        assert symbol_size[0] <= symbol_size[1]
        self.pixel = pixel
        self._dpi = 100
        self._figsize = pixel[0] / self._dpi, pixel[1] / self._dpi
        self.symbol_size = symbol_size
        self.symbol_alpha = symbol_alpha
        self.data = None
        self.specimen = None


    def set_data(
            self, data, test_size=None, test_alpha=None
            ):
        """ setter """
        self.data = data
        # plot
        if test_size is None:
            test_size = self.symbol_size[0]
        if test_alpha is None:
            test_alpha = self.symbol_alpha[0]
        self._test_view(data, test_size, test_alpha)

    
    def main(
            self, outdir:str="", name:str="", ratio:float=0.9,
            test_view:bool=True
            ):
        """
        dataをまとめてscatter arrayへと変換, npzで保存する
        input array, output arrayの順
        各arrayはsample, h, wの順
        time consuming, 1000回すのにXX min程度かかる
        
        Parameters
        ----------
        ratio: float, (0, 1)
            サンプルからデータ点を取得する割合

        """
        assert len(outdir) > 0
        assert (ratio > 0) & (ratio < 1) 
        # dataの準備
        array0 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        array1 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        for i, d in tqdm(enumerate(self.data)):
            # dataのhistogram化
            img0 = self.get_scatter_array(
                d, self.symbol_size[0], self.symbol_alpha[0]
                )
            img1 = self.get_scatter_array(
                d, self.symbol_size[1], self.symbol_alpha[1]
                )
            # imageの格納, grey scaleにしてチャンネルを潰すことに注意
            array0[i, :, :] = self.to_grey(img0)
            array1[i, :, :] = self.to_grey(img1)
        # NHWC形式になるように次元を追加
        array0 = array0[..., np.newaxis]
        array1 = array1[..., np.newaxis]
        # npzで保存
        if test_view:
            print("input example")
            self.imshow(array0[0, :, :, 0])
            print("output example")
            self.imshow(array1[0, :, :, 0])
        now = datetime.datetime.now().strftime('%Y%m%d')
        if len(name) > 0:
            fileout = outdir + SEP + f"dataset_{name}_{now}.npz"
        else:
            fileout = outdir + SEP + f"dataset_{now}.npz"
        np.savez_compressed(
            fileout, input=array0, output=array1, specimen=self.specimen
            )


    def get_scatter_array(self, data, size=30, alpha=0.5):
        """
        データをヒストグラムへと変換し, そのarrayを得る
        
        """
        # prepare histogram
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        # rangeを考慮しない方針
        ax.scatter(
            data[:, 0], data[:, 1],
            color="black", s=size, alpha=alpha
            )
        # convert array
        fig.canvas.draw() # レンダリング
        data = fig.canvas.tostring_rgb() # rgbのstringとなっている
        w, h = fig.canvas.get_width_height()
        c = len(data) // (w * h) # channelを算出
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
        plt.close()
        return img


    def to_grey(self, data):
        """ 得られたarrayをgrey scaleに変換する """
        # 輝度信号Yへと変換する
        # Y = 0.299 * R + 0.587 * G + 0.114 * B
        grey = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
        return grey


    def imshow(self, data, cmap='binary_r', figsize=None):
        """ show pixelized data """
        plt.figure(figsize=figsize)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        plt.imshow(data, aspect='equal', cmap=cmap)
        plt.show()

    
    def _test_view(self, data, test_size, test_alpha):
        """ refer to set_data """
        num = 4
        idx = list(range(len(data)))
        np.random.shuffle(idx)
        fig, axes = plt.subplots(1, num, figsize=(2.5 * num, 2.5))
        for i in range(num):
            axes[i].scatter(
                data[i][:, 0], data[i][:, 1],
                color="black", s=test_size, alpha=test_alpha
                )
        plt.tight_layout()
        plt.show()


class ContourMaker(DataMaker):
    """
    dataを読み込んで等高線図を表すnp配列へと変換する
    
    """
    def __init__(
            self, pixel:tuple=(64, 64), levels:tuple=(32, 16),
            ):
        super().__init__()
        assert levels[0] <= levels[1]
        self.pixel = pixel
        self._dpi = 100
        self._figsize = pixel[0] / self._dpi, pixel[1] / self._dpi
        self.levels = levels
        self.data = None
        self.specimen = None


    def set_data(
            self, data, test_levels=None
            ):
        """ setter """
        self.data = data
        # plot
        if test_levels is None:
            test_levels = self.levels[0]
        self._test_view(data, test_levels)


    def main(
            self, outdir:str="", name:str="", ratio:float=0.9,
            test_view:bool=True
            ):
        """
        dataをまとめてscatter arrayへと変換, npzで保存する
        input array, output arrayの順
        各arrayはsample, h, wの順
        time consuming, 1000回すのにXX min程度かかる
        
        Parameters
        ----------
        ratio: float, (0, 1)
            サンプルからデータ点を取得する割合

        """
        assert len(outdir) > 0
        assert (ratio > 0) & (ratio < 1) 
        # dataの準備
        array0 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        array1 = np.zeros((len(self.data), self.pixel[0], self.pixel[1]))
        for i, d in tqdm(enumerate(self.data)):
            # dataのhistogram化
            img0 = self.get_contour_array(
                d, self.levels[0]
                )
            img1 = self.get_contour_array(
                d, self.levels[1]
                )
            # imageの格納, grey scaleにしてチャンネルを潰すことに注意
            array0[i, :, :] = self.to_grey(img0)
            array1[i, :, :] = self.to_grey(img1)
        # NHWC形式になるように次元を追加
        array0 = array0[..., np.newaxis]
        array1 = array1[..., np.newaxis]
        # npzで保存
        if test_view:
            print("input example")
            self.imshow(array0[0, :, :, 0])
            print("output example")
            self.imshow(array1[0, :, :, 0])
        now = datetime.datetime.now().strftime('%Y%m%d')
        if len(name) > 0:
            fileout = outdir + SEP + f"dataset_{name}_{now}.npz"
        else:
            fileout = outdir + SEP + f"dataset_{now}.npz"
        np.savez_compressed(
            fileout, input=array0, output=array1, specimen=self.specimen
            )


    def get_contour_array(self, data, levels):
        """
        データを等高線図へと変換し, そのarrayを得る
        
        """
        # prepare histogram
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        ax = fig.add_subplot(1, 1, 1)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        # rangeを考慮しない方針
        ax.contourf(
            data[:, 0], data[:, 1],
            cmap="binary_r", levels=levels,
            )
        # convert array
        fig.canvas.draw() # レンダリング
        data = fig.canvas.tostring_rgb() # rgbのstringとなっている
        w, h = fig.canvas.get_width_height()
        c = len(data) // (w * h) # channelを算出
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)
        plt.close()
        return img


    def to_grey(self, data):
        """ 得られたarrayをgrey scaleに変換する """
        # 輝度信号Yへと変換する
        # Y = 0.299 * R + 0.587 * G + 0.114 * B
        grey = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
        return grey


    def imshow(self, data, cmap='binary_r', figsize=None):
        """ show pixelized data """
        plt.figure(figsize=figsize)
        plt.tick_params(
            labeltop=False, labelright=False, labelbottom=False, labelleft=False,
            top=False, right=False, bottom=False, left=False
            )
        plt.imshow(data, aspect='equal', cmap=cmap)
        plt.show()

    
    def _test_view(self, data, test_levels):
        """ refer to set_data """
        num = 4
        idx = list(range(len(data)))
        np.random.shuffle(idx)
        fig, axes = plt.subplots(1, num, figsize=(2.5 * num, 2.5))
        for i in range(num):
            axes[i].contourf(
                data[i][:, 0], data[i][:, 1],
                levels=test_levels, cmap="binary_r"
                )
        plt.tight_layout()
        plt.show()