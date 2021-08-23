import numpy as np
import pandas as pd
import math

import sklearn as skl
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

class Indicator:
    def __init__(self, initData: pd.DataFrame) -> None:
        ## 加载历史数据，构造Indicator对象，计算基于传入历史数据的各种指标
        ## 行：时间戳
        ## 列：每一个bar的字段信息（例如：ohlc，volume etc.）
        self.OriginalDatas = initData

    def SMA(self, periods=12):
        # 简单移动平均
        close = self.OriginalDatas.close
        ma = close.rolling_mean(window=periods, min_periods=2)
        return ma

    def EMA(self, periods=12):
        # 加权移动平均
        # periods大致等于希望用于计算的数据量，12就是12个bar
        # https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html

        ema = self.OriginalDatas.close.ewma(com=periods, min_periods=2)
        return ema

    def CCI(self, periods=12):
        # 顺势指标
        # <-100为超卖区间  >100为超买区间
        # https://zhuanlan.zhihu.com/p/82604762
        alpha = 0.015
        tp = self.OriginalDatas.loc[:,['high', 'low', 'close']].mean(axis=1)
        matp = tp.rolling_mean(window=periods, min_periods=periods)
        mdev = tp.rolling(window=periods, min_periods=periods).apply(lambda x: 
            (x-x.mean()).abs().mean())
        
        cci = (tp-matp)/(alpha*mdev)
        return cci


