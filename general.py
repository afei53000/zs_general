import datetime

import pandas as pd
import matplotlib as plt
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
# import MySQLdb

#数据处理块


def cleanData(asset,zb):

    minus = []
    
    # pd.DataFrame()
    # print(zb)

    for i in range(len(zb.index)):
        if (pd.isna(zb.iloc[i]) == False):
            # pdatas.flag[i] = 1
            k = zb.iloc[i]
            minus.append(k)

    # print(minus)
    minus = pd.DataFrame(minus)
    # print(minus)
    kk = minus.rolling(2, min_periods=2, axis=0).mean()
    minus= kk
    # print(minus)
    minus = minus - minus.shift(1)
    f = 0
    zb[0] = 0
    for i in range(1, len(zb.index)):
        if (pd.isna(zb[i]) == False):
            # pdatas.flag[i] = 1
            zb[i] = minus.iloc[f, 0]
            f = f + 1
    zb.index = zb.index + datetime.timedelta(days=26)  #

    pdatas = pd.concat([zb, asset], axis=1)
    # pdatas=pd.merge(zb,asset,how='outer')
    # pdatas = zb.join( asset,how='outer')

    pd.DataFrame(pdatas).columns = ['zb', 'asset']
    # print(pdatas)
    pdatas = pdatas[['zb', 'asset']]
    # 空值数据处理-----月～ri
    # zb_dropna= zb.dropna()
    # print(zb_dropna)
    for i in range(len(pdatas.index) - 1):
        # pdatas.zb[i] =np.nan
        if (pd.isna(pdatas.zb[i]) == False) & (pd.isna(pdatas.asset[i]) == True):
            # pdatas.flag[i] = 1
            pdatas.zb[i -1] = pdatas.zb[i]
            pdatas.zb[i]=np.nan

    pdatas = pdatas.dropna(subset=['zb'])
    # print(pdatas)

    # print(pdatas)
    return pdatas


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


    p1 = pd.DataFrame(pricein, columns=['datebuy', 'pricebuy'])
    p2 = pd.DataFrame(priceout, columns=['datesell', 'pricesell'])
    # p1/p2  是一个有2列的Dataframe  1：时间datebuy 2：买入/卖出价格pricebuy

    transactions = pd.concat([p1, p2], axis=1)
    pdatas['ret'] = pdatas.asset.pct_change(1).fillna(0)

    pdatas['nav'] = (1 + pdatas.ret * pdatas.position).cumprod()
    pdatas['benchmark'] = pdatas.asset / pdatas.asset[0]  # 错了

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

    N = 12

    # 年化收益率
    rety = strategy.nav[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1
    bench_rety = strategy.benchmark[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1

    # 夏普比
    Sharp = ((strategy.ret * strategy.position).mean()*N-0.026 )/( (strategy.ret * strategy.position).std() * np.sqrt(N))

    # 胜率
    VictoryRatio = ((transactions.pricesell - transactions.pricebuy) > 0).mean()

    DD = 1 - strategy.nav / strategy.nav.cummax()
    MDD = max(DD)

    nav_peryear = strategy.nav.groupby(strategy.year).last() / strategy.nav.groupby(strategy.year).first() - 1
    benchmark_peryear = strategy.benchmark.groupby(strategy.year).last() / strategy.benchmark.groupby(
        strategy.year).first() - 1

    excess_ret = nav_peryear - benchmark_peryear
    result_peryear = pd.concat([nav_peryear, benchmark_peryear, excess_ret], axis=1)
    result_peryear.columns = ['strategy_ret', 'bench_ret', 'excess_ret']
    result_peryear = result_peryear.T


    maxloss = min(transactions.pricesell / transactions.pricebuy - 1)

    result = {'Sharp': Sharp,
              'RetYearly': rety,
              'bench_rety': bench_rety,
              'WinRate': VictoryRatio,
              'MDD': MDD,
              'maxlossOnce': -maxloss,
              'num': round(strategy.flag.abs().sum() / strategy.shape[0], 1)}




    return result, result_peryear


choice=1
zhibiao_result=pd.DataFrame()

#载入数据
assets=pd.read_csv('zhishu.csv')
assets=assets.copy()
assets['date']=pd.to_datetime(assets['date'])
assets.set_index(assets['date'],inplace=True)
del assets['date']


hg_y=pd.read_csv('hg_yy.csv')
hg_y['date']=pd.to_datetime(hg_y['date'])
hg_y.set_index(hg_y['date'],inplace=True)
del hg_y['date']
hg_y=pd.DataFrame(hg_y)
# print(hg_y)
# print(assets)
# print(hg_y.iloc[:,1])
# print(assets.iloc[:,1])

sharp=pd.DataFrame()

################################################
m=0

for m in range(0,7):
    sharp = pd.DataFrame()
    rety = pd.DataFrame()
    bench_rety = pd.DataFrame()
    MDD = pd.DataFrame()
    VictoryRatio = pd.DataFrame()
    for i in range(m*10,
                   # 91):
                   min((m+1)*10),hg_y.shape[1]):

        one_hg=[]

        for k in range(0,
                       assets.shape[1]):
            # 1):
            zhibiao=hg_y.iloc[:,i].copy()
            asset=assets.iloc[:,k].copy()
            # print(zhibiao)
            # print(asset)
            pdatas=cleanData(asset,zhibiao)
            result=Strategy(pdatas)[0]
            # print(result)
            one_hg.append(result)
        one_hg = pd.DataFrame(one_hg)
        # zhibiao_result.append(one_hg)
        sharp['%s'%i]=one_hg['Sharp']
        bench_rety['%s' % i] = one_hg['bench_rety']
        rety['%s' % i] = one_hg['RetYearly']
        MDD['%s' % i] = one_hg['MDD']
        VictoryRatio['%s' % i] = one_hg['WinRate']

    VictoryRatio.to_csv('VictoryRatio%s.csv' % m)

    sharp.to_csv('sharp%s.csv'%m)
    bench_rety.to_csv('bench_rety%s.csv'%m)
    rety.to_csv('rety%s.csv'%m)
    MDD.to_csv('MDD%s.csv'%m)
    print('%s th is caculated' % m)

    #
    m=m+1

###################################################

####################################################
VictoryRatio_all=[]
sharp_all=[]
bench_rety_all=[]
rety_all=[]
MDD_all=[]
a=0
for a in range(0,7):
    x=pd.read_csv('bench_rety%s.csv'%a, header=None)#header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    bench_rety_all.append(x)
    y = pd.read_csv('rety%s.csv' % a, header=None)  # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    rety_all.append(y)
    z = pd.read_csv('MDD%s.csv' % a, header=None)  # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    MDD_all.append(z)
    v = pd.read_csv('sharp%s.csv' % a, header=None)  # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    sharp_all.append(v)
    w = pd.read_csv('VictoryRatio%s.csv' % a, header=None)  # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    VictoryRatio_all.append(w)

# print(bench_rety_all)
bench_rety_all=pd.concat(bench_rety_all,axis=1,ignore_index=True)
# print(bench_rety_all)
rety_all=pd.concat(rety_all,axis=1,ignore_index=True)
MDD_all=pd.concat(MDD_all,axis=1,ignore_index=True)
sharp_all=pd.concat(sharp_all,axis=1,ignore_index=True)
VictoryRatio_all=pd.concat(VictoryRatio_all,axis=1,ignore_index=True)
bench_ge=pd.DataFrame(['bench',00000])
rety_ge=pd.DataFrame(['rety',000])
MDD_ge=pd.DataFrame(['MDD',0000])
sharp_ge=pd.DataFrame(['sharp',000])
VictoryRatio_ge=pd.DataFrame(['Vct',000])
frames = [bench_ge,bench_rety_all,rety_ge,rety_all, MDD_ge,MDD_all,sharp_ge,sharp_all,VictoryRatio_ge, VictoryRatio_all]
kk=pd.concat(frames,axis=0,ignore_index=True)
kk.to_csv('final_zhishu.csv')


