import numpy as np
from numpy.lib.index_tricks import AxisConcatenator
import pandas as pd
import math

import sklearn as sk
import sklearn.linear_model as sklm
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

class Indicator:
    def __init__(self, initData: pd.DataFrame) -> None:
        ## 加载历史数据，构造Indicator对象，计算基于传入历史数据的各种指标
        ## 行：时间戳
        ## 列：每一个bar的字段信息（例如：ohlc，volume etc.）
        self.OriginalDatas = initData

    def SMA(self, periods=12, **kwargs):
        # 简单移动平均
        close = self.OriginalDatas.close
        ma = close.rolling(window=periods, min_periods=2).mean()
        return ma

    def EMA(self, periods=12, **kwargs):
        # 加权移动平均
        # periods大致等于希望用于计算的数据量，12就是12个bar
        # https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html
        if 'data' in kwargs.keys():
            ema = kwargs['data'].close.ewma(com=periods, min_periods=2)
        else:
            ema = self.OriginalDatas.close.ewma(com=periods, min_periods=2)
        return ema


    def CCI(self, periods=12, **kwargs):
        # 顺势指标
        # <-100为超卖区间  >100为超买区间
        # 要计算外部数据的CCI，传入kws，data:...
        # https://zhuanlan.zhihu.com/p/82604762
        alpha = 0.015
        tp = self.OriginalDatas.loc[:,['high', 'low', 'close']].mean(axis=1)
        matp = tp.rolling(window=periods, min_periods=periods).mean()
        mdev = tp.rolling(window=periods, min_periods=periods).apply(lambda x: 
            (x-x.mean()).abs().mean())
        
        cci = (tp-matp)/(alpha*mdev)
        return cci

    def MACD(self, long=26, short=12, periods=9, **kwargs):
        # 平滑异同移动平均
        # 利用中长期ema背离情况计算
        # 返回值： dif dea macd
        dif = self.EMA(short) - self.EMA(long)
        temp = {'data':dif}
        dea = self.EMA(periods, temp)
        macd = 2*(dif - dea)
        return dif, dea, macd

    def ACCER(self, periods=8, **kwargs):
        # 计算幅度涨速指标
        # 首先求出periods区间内close价格的斜率
        # 然后对当前价格做归一
        # 0一般为基准线，以上为上涨区间，以下为下跌区间
        
        close = self.OriginalDatas.close[-8:]
        linreg = sklm.LinearRegression()
        linreg.fit(range(periods), close)
        linreg.score(range(periods), close)
        accer = linreg.coef_/close[-1]
        return accer

    def BOLLINGER(self, periods=12, var=2, **kwargs):
        # 计算布林带，或称为保利加通道
        # 计算periods区间内的移动平均和标准差
        # 返回boll中轨，下轨，上轨bollMidd,bollDown,bollUp
        ma = self.SMA(periods)
        std = self.OriginalDatas.close.rolling(periods).std()
        bollUp = ma+var*std
        bollDown = ma-var*std
        bollMidd = ma
        return bollMidd,bollDown,bollUp

    def ADTM(self, periods=23, window=8, **kwargs):
        # 计算动态买卖气指标
        # 利用开盘价基准的上涨与下跌幅度判断标的关注度
        # periods是求adtm时使用的周期
        # window是计算adtm的sma时的周期
        # adtm在-1到1之间，高风险区间为0.5-1， 低风险区间为（-1）-（-0.5）
        # 关于adtm与它的sma之间的上下穿越关系和普通的一样。
        open = self.OriginalDatas.open
        preopen = open.shift(1)

        close = self.OriginalDatas.close
        preclose = close.shift(1)

        dtm = open
        dtm[open<=preopen] = 0
        dtm[open>preopen] = pd.DataFrame([self.OriginalDatas.high[open>preopen]-open[open>preopen], open[open>preopen]-preopen[open>preopen]]).max(axis=1)

        dbm = open
        dbm[open>=preopen] = 0
        dbm[open<preopen] = open[open<preopen] - self.OriginalDatas.low[open<preopen]

        stm = dtm.rolling(periods).sum()
        sbm = dbm.rolling(periods).sum()

        adtm = (stm-sbm)/pd.DataFrame([stm, sbm]).max(axis=1)
        adtmma = self.SMA(window, {'data':adtm})

        return adtm, adtmma

