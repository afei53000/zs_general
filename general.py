import datetime

import pandas as pd
import matplotlib as plt
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
# import MySQLdb

#数据处理块


def cleanData(asset,zb):
    # asset['date']=pd.to_datetime(asset['date'])
    # asset.set_index(asset['date'],inplace=True)
    # asset=asset['asset']



    # zb['date']=pd.to_datetime(zb['date'],format='%b-%y')

    # print(zb)

    # zb.set_index(zb['date'],inplace=True)


    # zb=zb.rolling(1,min_periods=4,axis=0).apply()
    # zb=zb[['zb','zb_roll']]
    #(
    minus=[]
        # pd.DataFrame()
    print(zb)

    for i in range(len(zb.index)):
        if (pd.isna(zb.iloc[i]) == False):
            # pdatas.flag[i] = 1
            k=zb.iloc[i]
            minus.append(k)

    # print(minus)
    minus=pd.DataFrame(minus)
    minus = minus - minus.shift(1)
    f=0
    zb[0] = 0
    for i in range(1,len(zb.index)):
        if (pd.isna(zb[i]) == False):
            # pdatas.flag[i] = 1
            zb[i]=minus.iloc[f,0]
            f=f+1
    zb.index=zb.index+datetime.timedelta(days=26)#





    pdatas=pd.concat([zb,asset],axis=1)
    # pdatas=pd.merge(zb,asset,how='outer')
    # pdatas = zb.join( asset,how='outer')

    pd.DataFrame(pdatas).columns=['zb','asset']
    # print(pdatas)
    pdatas = pdatas[['zb', 'asset']]
    # 空值数据处理-----周～月
    zb_dropna= zb.dropna()
    # print(zb_dropna)
    for i in range(len(pdatas.index)):
        # pdatas.zb[i] =np.nan
        if (pd.isna(pdatas.asset[i]) == False)&(pd.isna(pdatas.zb[i])==True):
            # pdatas.flag[i] = 1
            last_time=np.max(zb_dropna.index[zb_dropna.index<pdatas.index[i]])
            # np.max(last_time)
            # print(last_time)
            # print(pdatas.zb[pdatas.index==last_time])
            last_zb=pdatas[pdatas.index == last_time]
            if last_zb.empty==False:


                # print(last_zb.iloc[0,0])
                pdatas.zb[i] =last_zb.iloc[0,0]
                # pdatas[pdatas.index==last_time]
            pdatas[pdatas.index == last_time].zb=np.nan
    pdatas = pdatas.dropna(subset=['asset'])
    print(pdatas)

    # print(pdatas)
    return pdatas



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
# for i in range(len(pdatas.index)):
#     if (pd.isna(pdatas.asset[i])==False):
#         # pdatas.flag[i] = 1
#         pdatas.zb[i]=pdatas.zb[i-1]
#         pdatas.zb_roll[i]=pdatas.zb_roll[i-1]
# pdatas=pdatas.dropna(subset=['asset'])
# # print(pdatas)
# pdatas=pdatas[['zb','zb_roll','asset']]
# print(pdatas)

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
    # print( 'the transaction is :')
    # print(transactions)
    # transactions  是一个有2列的Dataframe  1：时间 2：买入和卖出价格

    # pdatas = pdatas.loc[max(0, win_long):, :].reset_index(drop=True)
    pdatas['ret'] = pdatas.asset.pct_change(1).fillna(0)

    # print(pdatas.iloc[:,1])#momentum
    # print(pdatas.iloc[:,1].pct_change(1).fillna(0))
    # print(pdatas.position)
    pdatas['nav'] = (1 + pdatas.ret * pdatas.position).cumprod()
    pdatas['benchmark'] = pdatas.asset / pdatas.asset[0]  # 错了
    # print(pdatas['benchmark'])
    # pdatas 是一个Dataframe 有8列 1：时间 2：宏观 3：价格CLOSE 4：position 5:flag 6.ret 7.nav 8.benchmark
    # print(pdatas.ret, pdatas.position)

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
    # xtick = np.round(np.linspace(0, strategy.shape[0] - 1, 7), 0).astype(int)
    # xticklabel = strategy.index[xtick].strftime("%y %b")

    # plt.figure(figsize=(9, 4))
    # ax1 = plt.axes()
    # plt.plot(np.arange(strategy.shape[0]), strategy.benchmark, 'black', label='benchmark', linewidth=2)
    # plt.plot(np.arange(strategy.shape[0]), strategy.nav, 'red', label='nav', linewidth=2)
    # # plt.plot(np.arange(strategy.shape[0]), strategy.nav / strategy.benchmark, 'orange', label='RS', linewidth=2)
    # # plt.plot(np.arange(strategy.shape[0]), 1 , 'grey', label='1', linewidth=1)
    #
    # plt.plot(np.arange(strategy.shape[0]), strategy.zb / 50000 + 1, 'orange', label='zb', linewidth=2)

    # plt.plot(np.arange(strategy.shape[0]), strategy.jtf_roll/50000+1 , 'blue', label='jtf_roll', linewidth=2)
    # lim = [1] * 120
    # plt.plot(lim, "r--")
    #
    # plt.legend()
    #
    # ax1.set_xticks(xtick)
    # ax1.set_xticklabels(xticklabel)
    # plt.savefig("/Users/feitongliu/Desktop/数据/jtf_momentum.png", dpi=100)
    # plt.show()

    maxloss = min(transactions.pricesell / transactions.pricebuy - 1)

    # print('------------------------------')
    # print('夏普比为:', round(Sharp, 2))
    # print('年化收益率为:{}%'.format(round(rety * 100, 2)))
    # print('benchmark年化收益率为:{}%'.format(round(bench_rety * 100, 2)))
    # print('胜率为：{}%'.format(round(VictoryRatio * 100, 2)))
    # print('最大回撤率为：{}%'.format(round(MDD * 100, 2)))
    # # print('单次最大亏损为:{}%'.format(round(-maxloss * 100, 2)))
    # print('月均交易次数为：{}(买卖合计)'.format(round(strategy.flag.abs().sum() / strategy.shape[0] * 20, 2)))

    result = {'Sharp': Sharp,
              'RetYearly': rety,
              'WinRate': VictoryRatio,
              'MDD': MDD,
              'maxlossOnce': -maxloss,
              'num': round(strategy.flag.abs().sum() / strategy.shape[0], 1)}

    # result = pd.DataFrame.from_dict(result, orient='index').T
    # result = pd.DataFrame.from_dict(result, orient='index')

    # print(result)


    return result, result_peryear


choice=1
zhibiao_result=pd.DataFrame()

#载入数据
assets=pd.read_csv('berra.csv')
assets=assets.copy()
assets['date']=pd.to_datetime(assets['date'])
assets.set_index(assets['date'],inplace=True)
del assets['date']


hg_y=pd.read_csv('hg_yue.csv')
hg_y['date']=pd.to_datetime(hg_y['date'])
hg_y.set_index(hg_y['date'],inplace=True)
del hg_y['date']
hg_y=pd.DataFrame(hg_y)
# print(hg_y)
# print(assets)
# print(hg_y.iloc[:,1])
# print(assets.iloc[:,1])

# for i in range(0,hg_y.shape[1]-1):
#     for k in range(0,assets.shape[1]-1):
sharp=pd.DataFrame()
# rety=pd.DataFrame()
# VictoryRatio=pd.DataFrame()
# MDD=pd.DataFrame()
# -maxloss=pd.DataFrame()
# =pd.DataFrame()


for i in range(0,
               30):
              # 1):

    one_hg=[]
    for k in range(0,
                   # assets.shape[1]):
        1):
        zhibiao=hg_y.iloc[:,i]
        asset=assets.iloc[:,k]
        # print(zhibiao)
        # print(asset)
        pdatas=cleanData(asset,zhibiao)
        result=Strategy(pdatas)[0]
        # print(result)
        one_hg.append(result)
    one_hg = pd.DataFrame(one_hg)
    # zhibiao_result.append(one_hg)
    sharp['%s'%i]=one_hg['Sharp']
    # rety['%s' % i] = one_hg['rety']
    # VictoryRatio['%s' % i] = one_hg['VictoryRatio']
    # MDD['%s' % i] = one_hg['MDD']
    # -maxloss['%s' % i] = one_hg['-maxloss']
    # sharp.append(one_hg['Sharp'])
# print(one_hg)
sharp.to_csv('sharp13.csv')


# rety.to_csv('rety5.csv')
print(sharp)
    # one_row=pd.DataFrame(one_hg)
    # print(one_row)
#     zhibiao_result.append(one_row)
# # zhibiao_1=pd.DataFrame(zhibiao_result)
# # Sharp=zhibiao_1['Sharp']
# pd.DataFrame(zhibiao_result).to_csv('Sharp.csv')
# print(zhibiao_result)



# a=pd.read_csv('sharp1.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# b=pd.read_csv('sharp2.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# c=pd.read_csv('sharp3.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# d=pd.read_csv('sharp4.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# e=pd.read_csv('sharp5.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# f=pd.read_csv('sharp6.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# g=pd.read_csv('sharp7.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# h=pd.read_csv('sharp8.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# i=pd.read_csv('sharp9.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# j=pd.read_csv('sharp10.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# k=pd.read_csv('sharp11.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# l=pd.read_csv('sharp12.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
# m=pd.read_csv('sharp13.csv', header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
#
# all= pd.concat([a,b,c,d,e,f,g,h,i,j,k,l,m],axis=1,ignore_index=True)
# all.to_csv('sharpp.csv',index=False) #header=0表示不保留列名，index=False表示不保留行索引，mode='a'表示附加方式写入，文件原有内容不会被清除



# b.to_csv('sharpp.csv', mode='a', index=True, header=0)
# c.to_csv('sharpp.csv', mode='a', index=True, header=0)
# d.to_csv('sharpp.csv', mode='a', index=True, header=0)
# e.to_csv('sharpp.csv', mode='a', index=True, header=0)
# f.to_csv('sharpp.csv', mode='a', index=True, header=0)
# g.to_csv('sharpp.csv', mode='a', index=True, header=0)
# h.to_csv('sharpp.csv', mode='a', index=True, header=0)
# i.to_csv('sharpp.csv', mode='a', index=True, header=0)
# j.to_csv('sharpp.csv', mode='a', index=True, header=0)
# k.to_csv('sharpp.csv', mode='a', index=True, header=0)
# l.to_csv('sharpp.csv', mode='a', index=True, header=0)
# m.to_csv('sharpp.csv', mode='a', index=True, header=0)

