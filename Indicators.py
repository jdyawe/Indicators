from typing import OrderedDict
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import math

import sklearn as sk
import sklearn.linear_model as sklm
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

'''
部分指标的意义及用法见
http://www.10jqka.com.cn/ex_info/man_html/f4.htm
'''

class Indicator:
    '''
    加载历史数据，构造Indicator对象，计算基于传入历史数据的各种指标\n
    行：时间戳\n
    列：每一个bar的字段信息（例如：ohlc，volume etc.）\n
    时间戳由上到下为历史到今天！！这个很重要，在下面所有信号的计算中几乎都用到了这个默认规定
    '''

    def __init__(self, initData: pd.DataFrame) -> None:
        
        self.OriginalDatas = initData

    def SMA(self, periods=12, **kwargs):
        '''
        简单移动平均
        '''
        close = self.OriginalDatas.close
        ma = close.rolling(window=periods, min_periods=2).mean()
        return ma

    def EMA(self, periods=12, **kwargs):
        '''
        加权移动平均\n
        periods大致等于希望用于计算的数据量，12就是12个bar\n
        https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.ewma.html
        '''

        ema = self.OriginalDatas.close.ewma(com=periods, min_periods=2)
        # 衰减系数：
        # alpha = 1/(1+com)
        # alpha = 2/(1+span)
        return ema


    def CCI(self, periods=12, **kwargs):
        '''
        顺势指标\n
        CCI<-100为超卖区间, CCI>100为超买区间\n
        要计算外部数据的CCI，传入kws，data:...\n
        https://zhuanlan.zhihu.com/p/82604762
        '''
        
        alpha = 0.015
        tp = self.OriginalDatas.loc[:,['high', 'low', 'close']].mean(axis=1)
        matp = tp.rolling(window=periods, min_periods=periods).mean()
        mdev = tp.rolling(window=periods, min_periods=periods).apply(lambda x: 
            (x-x.mean()).abs().mean())
        
        cci = (tp-matp)/(alpha*mdev)
        return cci

    def MACD(self, long=26, short=12, periods=9, **kwargs):
        '''
        平滑异同移动平均\n
        利用中长期ema背离情况计算\n
        返回值： dif dea macd
        '''
        dif = self.EMA(short) - self.EMA(long)
        temp = {'data':dif}
        dea = self.EMA(periods, temp)
        macd = 2*(dif - dea)
        return dif, dea, macd

    def ACCER(self, periods=8, **kwargs):
        '''
        计算幅度涨速指标\n
        首先求出periods区间内close价格的斜率\n
        然后对当前价格做归一\n
        0一般为基准线，以上为上涨区间，以下为下跌区间
        '''
        
        close = self.OriginalDatas.close[-8:]
        linreg = sklm.LinearRegression()
        linreg.fit(range(periods), close)
        linreg.score(range(periods), close)
        accer = linreg.coef_/close[-1]
        return accer

    def BOLLINGER(self, periods=12, var=2, **kwargs):
        '''
        计算布林带，或称为保利加通道\n
        计算periods区间内的移动平均和标准差\n
        返回boll中轨，下轨，上轨bollMidd,bollDown,bollUp
        '''
        ma = self.SMA(periods)
        std = self.OriginalDatas.close.rolling(periods).std()
        bollUp = ma+var*std
        bollDown = ma-var*std
        bollMidd = ma
        return bollMidd,bollDown,bollUp

    def ADTM(self, periods=23, window=8, **kwargs):
        '''
        计算动态买卖气指标\n
        利用开盘价基准的上涨与下跌幅度判断标的关注度\n
        periods是求adtm时使用的周期\n
        window是计算adtm的sma时的周期\n
        adtm在-1到1之间，高风险区间为0.5-1， 低风险区间为（-1）-（-0.5）\n
        关于adtm与它的sma之间的上下穿越关系和普通的一样。
        '''
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

    def RSI(self, periods=12, **kwargs):
        '''
        计算相对强弱指标\n
        在rsi>50范围内为强势区间， <50为弱势区间\n
        跌破20为超卖区间，涨破80为超买区间
        '''
        close = self.OriginalDatas.close
        change = close.diff(1)
        up = change.rolling(periods).apply(lambda x: (x>0).sum())
        down = change.rolling(periods).apply(lambda x: (x<0).sum())
        rsi = up/(up+down)*100

        return rsi        

    def ATR(self, periods=12, **kwargs):
        '''
        计算平均波动幅度指标，也称为均幅指标\n
        计算简单的波动率描述
        '''
        tr = pd.DataFrame()
        tr['hl'] = self.OriginalDatas.high-self.OriginalDatas.low
        tr['hc'] = (self.OriginalDatas.high-self.OriginalDatas.close.shift(1)).abs()
        tr['lc'] = (self.OriginalDatas.low-self.OriginalDatas.close.shift(1)).abs()
        tr['tr'] = tr.max(axis=1)
        atr = tr['tr'].rolling(window=periods).mean()
        return atr

    def KDJ(self, periods=2, window=9, **kwargs):
        '''
        计算随即指数\n
        KD波动位于0-100之间，>80为超买，<20为超卖\n
        J>100为超买，<0为超卖\n
        返回值为(k,d,j)
        '''
        close = self.OriginalDatas.close
        ndaylow = self.OriginalDatas.low.rolling(window).min()
        ndayhigh = self.OriginalDatas.high.rolling(window).max()
        rsv = (close-ndaylow)/(ndayhigh-ndaylow)*100

        k = rsv.ewma(com=periods)
        d = k.ewma(com=periods)
        j = 3*k-2*d
        # 此处的平滑移动平均也可以用移动平均代替
        # 具体效果不明
        return k,d,j


    def SAR(self, periods=4):
        '''
        计算抛物线指标\n
        首先选取前periods天判断上升与下降趋势\n
        这个周期最后一天的收盘价在这个周期的高低点价位处于60%之上为上升，否则为下降\n
        下面以初始周期为上升为例，下降相反\n
        当天的SAR设置为前periods日内最低价，极值点为周期内最高价\n
        SARt = SARt-1 + AF*(EPt-1 - SARt-1)\n
        AF初始值为0.02，若某日更新了极值点价格，则AF增加0.02，最大0.2\n
        若SARt比当日最低点高，则出现反转，当日SARt应该设置为前periods最高价，进入下跌阶段计算
        '''
        data = self.OriginalDatas.loc[:,['close', 'high', 'low']]
        sar = pd.Series([np.nan]*len(data))
        AF = 0.02

        # initialization
        if data.close[periods-1] > data.high[:periods].max()*0.6 + data.low[:periods].min()*0.4:
            sar[periods-1] = data.low[:periods].min()
            Flag = 'up'
        else:
            sar[periods-1] = data.high[:periods].max()
            Flag = 'down'
        
        # step forward
        for date in range(periods, data.shape[1]):
            if Flag == 'up':
                EP = self.OriginalDatas.high[date-periods:date].max()
                # 计算上升sar
                if self.OriginalDatas.high[date] > EP:
                    AF = min([AF+0.02, 0.2])
                
                sar[date] = sar[date-1] + AF*(EP - sar[date-1])

                if self.OriginalDatas.low[date]<=sar[date]:
                    sar[date] = self.OriginalDatas.high[date-periods+1:date+1].max()
                    Flag = 'down'
                    AF = 0.02

            else:
                EP = self.OriginalDatas.low[date-periods:date].min()
                # 计算下降sar
                if self.OriginalDatas.low[date] < EP:
                    AF = min([AF+0.02, 0.2])
                
                sar[date] = sar[date-1] + AF*(EP - sar[date-1])

                if self.OriginalDatas.high[date]>=sar[date]:
                    sar[date] = self.OriginalDatas.low[date-periods+1:date+1].min()
                    Flag = 'up'
                    AF = 0.02

        return sar

    def WR(self, periods=14):
        '''
        计算威廉指标，震荡指标\n
        判断当前情况的超买与超卖\n
        指标在0-100\n
        >80超卖， <20超买， 和RSI指标使用方法相反\n
        判断的大概是当前收盘价和过去periods周期中最高与最低价的位置
        '''
        close = self.OriginalDatas.close
        high = self.OriginalDatas.high
        low = self.OriginalDatas.low

        high = high.rolling(window=periods).apply(max)
        low = low.rolling(window=periods).apply(min)

        wr = (high-close)/(high-low)*100

        return wr

    def ROC(self, retard=8):
        '''
        计算变动率指标 rate of change\n
        对于不同的标的需要自己判断ROC的不同的范围\n
        用于判断当前股价变动动力大小
        '''
        roc = self.OriginalDatas.close/self.OriginalDatas.close.shift(retard)
        return roc

    def OSC(self, periods):
        '''
        计算震荡指标\n
        反应当前价格与一段时间内均价的偏离值\n
        一段时间内的均价可以直接取收盘价，也可以是ohlc的均值，此处取后者
        '''

        # baseprice = self.OriginalDatas.close.rolling(window=periods).apply(mean)

        baseprice = 2*self.OriginalDatas.close + self.OriginalDatas.high + self.OriginalDatas.low
        baseprice = baseprice/4
        baseprice = baseprice.rolling(window=periods).apply(mean)

        osc = self.OriginalDatas.close/baseprice

        return osc
    
    def BIAS(self, periods=12):
        '''
        计算价格乖离率\n
        乖离率越高，回复均价的概率越大\n
        简单均价可以被看作过去一段时间内的成本价均价
        '''

        close = self.OriginalDatas.close.rolling(window=periods).apply(mean)
        bias = (self.OriginalDatas.close-close)/close*100

        return bias

    def UDL(self, p1=3, p2=5, p3=10, p4=20):
        '''
        计算引力线指标\n
        股票走势价格会回归于价值\n
        当udl过高时倾向于价格下行，过低时倾向于价格上行
        '''

        close = self.OriginalDatas.close
        c1 = close.rolling(window=p1).apply(mean)
        c2 = close.rolling(window=p2).apply(mean)
        c3 = close.rolling(window=p3).apply(mean)
        c4 = close.rolling(window=p4).apply(mean)

        udl = (c1+c2+c3+c4)/4
        return udl
                    
    def VR(self, periods=12):
        '''
        计算成交量变异率 (Volume Ratio)\n
        用于描述成交量的强弱\n
        [40,70]:低价区域，交易稀少，人气涣散，可能是价格底部区间\n
        [80,150]: 安全区域，买卖盘开始增多，人气积聚\n
        [160,450]: 获利区域，买盘动能强大，看涨情绪高涨\n
        [450,]: 警戒区域，标的超买情况严重，当前表现超过了一般情况，没有大的利好消息很难维持这样的涨势
        '''
        
        close = self.OriginalDatas.loc[:, ['close', 'volume']]
        close['change'] = close.close.diff(1)
        
        vr = close.rolling(periods).apply(lambda x: x.volume[x.change>0].sum()/x.volume[x.change<0].sum())
        return vr

    def CR(self,periods=12):
        '''
        计算中间意愿指标\n
        描述多空双方力量对比\n
        计算某一周期内最高价、最低价与前一日日内中间价价差的和\n
        cr指标不同周期的均线会构成博弈区间，在区间内博弈会加剧，区间外情绪较为一致\n
        周期越长，参与人员越多，博弈区间越激烈，所形成的压力或支撑作用越强
        '''

        baseprice = 2*self.OriginalDatas.close + self.OriginalDatas.high + self.OriginalDatas.low
        baseprice = baseprice.shift(1)/4 # 代表中间价

        p1 = (self.OriginalDatas.high-baseprice).rolling(periods).apply(lambda x: x[x>0].sum())
        p2 = (baseprice-self.OriginalDatas.low).rolling(periods).apply(lambda x: x[x>0].sum())

        cr = p1/p2*100
        return cr

    def ARBR(self,periods=12):
        '''
        计算人气指标与买卖意愿指标\n
        CR指标的补充性指标\n
        AR指标为(H-O)/(O-L)\n
        BR指标为(H-CY)/(CY-L)
        '''

        high = self.OriginalDatas.high
        open = self.OriginalDatas.open
        low = self.OriginalDatas.low
        close = self.OriginalDatas.close.shift(1)
        
        ar = (high-open).rolling(window=periods).apply(sum)/(open-low).rolling(window=periods).apply(sum)
        br = (high-close).rolling(window=periods).apply(sum)/(close-low).rolling(window=periods).apply(sum)

        return ar, br

    def VSTD(self,):
        pass

    def OBV(self,):
        pass

    def PVT(self,):
        pass

    def CDP(self,):
        pass

    def MIKE(self,):
        pass

    def DDI(self,):
        pass

    def RCCD(self,):
        pass

    def MI(self,):
        pass

    def UOS(self,):
        pass

    def PBX(self,):
        pass

    def OBOS(self,):
        pass

    def ADR(self,):
        pass

    