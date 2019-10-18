import datetime

import pandas as pd
import matplotlib as plt
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
# import MySQLdb

#数据处理块


def cleanData(asset,zb):
    asset['date']=pd.to_datetime(asset['date'])
    asset.set_index(asset['date'],inplace=True)
    asset=asset['asset']
    # print(mo)


    # jtf["date"]=jtf["date"].astype('datetime64')
    # jtf["jtf"]=jtf["jtf"].astype('float64')

    zb['date']=pd.to_datetime(zb['date'],format='%b-%y')
    # jtf.set_index("date", inplace=True)
    print(zb)

    zb.set_index(zb['date'],inplace=True)
    zb['zb_roll']=zb['zb'].rolling(4,min_periods=4,axis=0).mean()
    zb=zb[['zb','zb_roll']]
    # print(jtf)
    # jtf.plot()
    # plt.show()
    # jtf.set_index(jtf['date'],inplace=True)
    # jtf=jtf['jtf']
# print(jtf)
    pdatas=pd.concat([zb,asset],axis=1)
    pd.DataFrame(pdatas).columns=['zb','zb_roll','asset']
    print(pdatas)
    return pdatas


#载入数据
mom=pd.read_csv('berra.csv')
mom=mom.copy()
asset=mom[['date','momentum']]
pd.DataFrame(asset).columns=['date','asset']


jtf=pd.read_csv('pmi.csv')
jtf=jtf.copy()

zb=jtf[['date','jiyadingdan']]
pd.DataFrame(zb).columns=['date','zb']
zb['zb'] = zb['zb'] - zb['zb'].shift(-1)
choice=-1 #positive 买入资产

pdatas=cleanData(asset,zb)
# zb=jtf.copy()
# zb['syl']=zb['syl']-zb['syl'].shift(-1)
# pdatas=cleanData(asset,zb)
#数据处理块
# mom=pd.read_csv('hs300.csv')
# mom=mom.copy()
# asset=mom[['date','hs300']]
# pd.DataFrame(asset).columns=['date','asset']
#
#
# jtf=pd.read_csv('jtf.csv')
# jtf=jtf.copy()
# pd.DataFrame(jtf).columns=['date','zb']
#
# zb=jtf.copy()
# pdatas=cleanData(asset,zb)

#空值数据处理-----季～天 改为0
# for i in range(len(pdatas.index)):
#     if pd.isna(pdatas.zb[i])==True:
#         # pdatas.flag[i] = 1
#         pdatas.zb[i]=0
#
# pdatas=pdatas.dropna(subset=['asset'])
# # print(pdatas)
# pdatas=pdatas[['zb','zb_roll','asset']]
# print(pdatas)

# #空值数据处理-----季～天
# for i in range(len(pdatas.index)):
#     if (pd.isna(pdatas.zb[i-1])==False)&(pd.isna(pdatas.zb[i])==True):
#         # pdatas.flag[i] = 1
#         pdatas.zb[i]=pdatas.zb[i-1]
#     if (pd.isna(pdatas.zb_roll[i - 1]) == False) & (pd.isna(pdatas.zb_roll[i]) == True):
#         pdatas.zb_roll[i]=pdatas.zb_roll[i-1]
# pdatas=pdatas.dropna(subset=['asset'])
# # print(pdatas)
# pdatas=pdatas[['zb','zb_roll','asset']]
# print(pdatas)


# 空值数据处理-----周～天
# for i in range(len(pdatas.index)):
#     if (pd.isna(pdatas.zb[i-1])==False)&(pd.isna(pdatas.zb[i])==True):
#         # pdatas.flag[i] = 1
#         pdatas.zb[i]=pdatas.zb[i-1]
#     if (pd.isna(pdatas.zb_roll[i - 1]) == False) & (pd.isna(pdatas.zb_roll[i]) == True):
#         pdatas.zb_roll[i]=pdatas.zb_roll[i-1]
# pdatas=pdatas.dropna(subset=['asset'])
# # print(pdatas)
# pdatas=pdatas[['zb','zb_roll','asset']]
# print(pdatas)



#空值数据处理-----周～月
for i in range(len(pdatas.index)):
    if (pd.isna(pdatas.asset[i])==False):
        # pdatas.flag[i] = 1
        pdatas.zb[i]=pdatas.zb[i-1]
        pdatas.zb_roll[i]=pdatas.zb_roll[i-1]
pdatas=pdatas.dropna(subset=['asset'])
# print(pdatas)
pdatas=pdatas[['zb','zb_roll','asset']]
print(pdatas)

def Strategy(pdatas):
    # pdatas = datas.copy();win_long = 12;win_short = 6;lossratio = 999;
    """
    pma：计算均线的价格序列
    win:窗宽
    lossratio：止损率,默认为0
    """

    pdatas = pdatas.copy()

    pdatas['position'] = 0  # 记录持仓
    pdatas['flag'] = 0  # 记录买卖
    # pdatas 是一个Dataframe 有5列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag 6.rolling
    pricein = []
    priceout = []
    price_in = 1



    ########pdatas['rolling']=pdatas.iloc[:,2].rolling(win_short,min_periods=2,axis=0).mean()
    if choice==1:

        for i in range(0, pdatas.shape[0] - 1):

            # 仅多择时策略：仅在出现状态 b 的时候做多资产 C，验证得到的结果称为正向关系显著；
            # 当前出现状态b（b状态：宏观指标下降），做多
            # (pd.isna(pdatas.momentum[i])==False)&
            if (pdatas.zb[i]>0):
                # &(pdatas.rolling[i]>0)
                pdatas.flag[i] = 1
                pdatas.position[i+1] = 1
                # pdatas 是一个Dataframe 有5列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag

                date_in = pdatas.index[i]
                price_in = pdatas.asset[i]
                pricein.append([date_in, price_in])
            if (pdatas.zb[i] <=0):
                # &(pdatas.rolling[i]>0)
                pdatas.flag[i] = -1
                pdatas.position[i+1] = 0
                # pdatas 是一个Dataframe 有5列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag

                date_out = pdatas.index[i]
                price_out = pdatas.asset[i]
                priceout.append([date_out, price_out])
    elif choice == -1:
        for i in range(0, pdatas.shape[0] - 1):

            # 仅多择时策略：仅在出现状态 b 的时候做多资产 C，验证得到的结果称为正向关系显著；
            # 当前出现状态b（b状态：宏观指标下降），做多
            # (pd.isna(pdatas.momentum[i])==False)&
            if (pdatas.zb[i] < 0):
                # &(pdatas.rolling[i]>0)
                pdatas.flag[i] = 1
                pdatas.position[i + 1] = 1
                # pdatas 是一个Dataframe 有5列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag

                date_in = pdatas.index[i]
                price_in = pdatas.asset[i]
                pricein.append([date_in, price_in])
            if (pdatas.zb[i] >= 0):
                # &(pdatas.rolling[i]>0)
                pdatas.flag[i] = -1
                pdatas.position[i + 1] = 0
                # pdatas 是一个Dataframe 有5列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag

                date_out = pdatas.index[i]
                price_out = pdatas.asset[i]
                priceout.append([date_out, price_out])

        
        # if (pdatas.jtf[i] > 0):
        #     # &(pdatas.rolling[i]>0)
        #     pdatas.flag[i] = -1
        #     pdatas.position[i + 1] = 0
        #     # pdatas 是一个Dataframe 有5列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag
        #
        #     date_in = pdatas.index[i]
        #     price_in = pdatas.momentum[i]
        #     pricein.append([date_in, price_in])
    # pricein/priceout  是一个有2列的列表 1：时间 2：买入/卖出价格

    # 当前持仓，下跌超出止损率，止损
    # elif (pdatas.position[i] == 1)& (((pdatas.CLOSE[i] / price_in )- 1 )< -lossratio):
    #             pdatas.loc[i, 'flag'] = -1
    #             pdatas.loc[i + 1, 'position'] = 0
    #
    #             priceout.append([pdatas.DateTime[i], pdatas.loc[i, 'CLOSE']])
    #
    #         # 当前持仓，死叉，平仓
    # elif (pdatas.sma[i - 1] > pdatas.lma[i - 1]) & (pdatas.sma[i] < pdatas.lma[i]) & (pdatas.position[i] == 1):
    #             pdatas.loc[i, 'flag'] = -1
    #             pdatas.loc[i + 1, 'position'] = 0
    #
                # priceout.append([pdatas.DateTime[i], pdatas.loc[i, 'CLOSE']])
    #
    #         # 其他情况，保持之前仓位不变
    # else:
    #             pdatas.loc[i + 1, 'position'] = pdatas.loc[i, 'position']

    p1 = pd.DataFrame(pricein, columns=['datebuy', 'pricebuy'])
    p2 = pd.DataFrame(priceout, columns=['datesell', 'pricesell'])
    # p1/p2  是一个有2列的Dataframe  1：时间datebuy 2：买入/卖出价格pricebuy

    transactions = pd.concat([p1, p2], axis=1)
    print( 'the transaction is :')
    print(transactions)
    # transactions  是一个有2列的Dataframe  1：时间 2：买入和卖出价格

    # pdatas = pdatas.loc[max(0, win_long):, :].reset_index(drop=True)
    pdatas['ret'] = pdatas.asset.pct_change(1).fillna(0)

    # print(pdatas.iloc[:,1])#momentum
    # print(pdatas.iloc[:,1].pct_change(1).fillna(0))
    # print(pdatas.position)
    pdatas['nav'] = (1 + pdatas.ret * pdatas.position).cumprod()
    pdatas['benchmark'] = pdatas.asset / pdatas.asset[0]  # 错了
    print(pdatas['benchmark'])
    # pdatas 是一个Dataframe 有8列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag 6.ret 7.nav 8.benchmark
    print(pdatas.ret, pdatas.position)

    df = pd.DataFrame(pdatas.index.strftime('%Y-%m-%d').str.split('-').tolist(),
                      columns=['year', 'month', 'day'], dtype=int)

    pdatas['ii'] = np.arange(pdatas.shape[0])
    pdatas['da'] = pdatas.index
    pdatas.set_index(pdatas['ii'], inplace=True)
    # print(pdatas)
    # print(df)
    pdatas = pd.concat([pdatas, df], axis=1)
    pdatas.set_index(pdatas['da'], inplace=True)

    pdatas.to_csv('result.csv')

    stats, result_peryear = performace(transactions, pdatas)

    return stats, result_peryear, transactions, pdatas


def performace(transactions, strategy):
    # strategy = pdatas.copy();
    # strategy 是一个Dataframe 有8列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag 6.ret 7.nav 8.benchmark

    N = 250

    # 年化收益率
    rety = strategy.nav[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1
    bench_rety = strategy.benchmark[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1

    # 夏普比
    Sharp = (strategy.ret * strategy.position).mean() / (strategy.ret * strategy.position).std() * np.sqrt(N)

    # 胜率
    VictoryRatio = ((transactions.pricesell - transactions.pricebuy) > 0).mean()

    DD = 1 - strategy.nav / strategy.nav.cummax()
    MDD = max(DD)

    # 策略逐年表现??????
    # strategy['date']=strategy.index.copy()
    # strategy['date']=pd.DataFrame(strategy['date']).to_datetime()
    #
    # # strategy['year'] = strategy.DateTime.apply(lambda x:x[:4])
    nav_peryear = strategy.nav.groupby(strategy.year).last() / strategy.nav.groupby(strategy.year).first() - 1
    benchmark_peryear = strategy.benchmark.groupby(strategy.year).last() / strategy.benchmark.groupby(
        strategy.year).first() - 1

    excess_ret = nav_peryear - benchmark_peryear
    result_peryear = pd.concat([nav_peryear, benchmark_peryear, excess_ret], axis=1)
    result_peryear.columns = ['strategy_ret', 'bench_ret', 'excess_ret']
    result_peryear = result_peryear.T

    # 作图
    xtick = np.round(np.linspace(0, strategy.shape[0] - 1, 7), 0).astype(int)
    xticklabel = strategy.index[xtick].strftime("%y %b")

    plt.figure(figsize=(9, 4))
    ax1 = plt.axes()
    plt.plot(np.arange(strategy.shape[0]), strategy.benchmark, 'black', label='benchmark', linewidth=2)
    plt.plot(np.arange(strategy.shape[0]), strategy.nav, 'red', label='nav', linewidth=2)
    # plt.plot(np.arange(strategy.shape[0]), strategy.nav / strategy.benchmark, 'orange', label='RS', linewidth=2)
    # plt.plot(np.arange(strategy.shape[0]), 1 , 'grey', label='1', linewidth=1)

    plt.plot(np.arange(strategy.shape[0]), strategy.zb / 50000 + 1, 'orange', label='zb', linewidth=2)
    # plt.plot(np.arange(strategy.shape[0]), strategy.asset, 'blue', label='asset', linewidth=2)

    # plt.plot(np.arange(strategy.shape[0]), strategy.jtf_roll/50000+1 , 'blue', label='jtf_roll', linewidth=2)
    lim = [1] * 120
    plt.plot(lim, "r--")

    plt.legend()

    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xticklabel)
    plt.savefig("/Users/feitongliu/Desktop/数据/jtf_momentum.png", dpi=100)
    plt.show()

    maxloss = min(transactions.pricesell / transactions.pricebuy - 1)
    print('------------------------------')
    print('夏普比为:', round(Sharp, 2))
    print('年化收益率为:{}%'.format(round(rety * 100, 2)))
    print('benchmark年化收益率为:{}%'.format(round(bench_rety * 100, 2)))
    print('胜率为：{}%'.format(round(VictoryRatio * 100, 2)))
    print('最大回撤率为：{}%'.format(round(MDD * 100, 2)))
    # print('单次最大亏损为:{}%'.format(round(-maxloss * 100, 2)))
    print('月均交易次数为：{}(买卖合计)'.format(round(strategy.flag.abs().sum() / strategy.shape[0] * 20, 2)))

    result = {'Sharp': Sharp,
              'RetYearly': rety,
              'WinRate': VictoryRatio,
              'MDD': MDD,
              'maxlossOnce': -maxloss,
              'num': round(strategy.flag.abs().sum() / strategy.shape[0], 1)}

    result = pd.DataFrame.from_dict(result, orient='index').T


    return result, result_peryear


Strategy(pdatas)