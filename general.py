import datetime
import pandas as pd
import matplotlib as plt
import numpy as np
from matplotlib import pyplot as plt


#数据处理块，清洗空值、补齐面板数据、移动平均、变化值、实际可得数据调整
def cleanData(asset,zb):
    minus = []
    for i in range(len(zb.index)):
        if (pd.isna(zb.iloc[i]) == False):
            # pdatas.flag[i] = 1
            k = zb.iloc[i]
            minus.append(k)
    minus = pd.DataFrame(minus)

    kk = minus.rolling(2, min_periods=2, axis=0).mean()
    minus= kk

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
    pd.DataFrame(pdatas).columns = ['zb', 'asset']
    pdatas = pdatas[['zb', 'asset']]
    # 空值数据处理-----月～日
    for i in range(len(pdatas.index) - 1):
        # pdatas.zb[i] =np.nan
        if (pd.isna(pdatas.zb[i]) == False) & (pd.isna(pdatas.asset[i]) == True):
            # pdatas.flag[i] = 1
            pdatas.zb[i -1] = pdatas.zb[i]
            pdatas.zb[i]=np.nan

    pdatas = pdatas.dropna(subset=['zb'])
    return pdatas


def Strategy(pdatas):
    pdatas = pdatas.copy()

    pdatas['position'] = 0  # 记录持仓
    pdatas['flag'] = 0  # 记录买卖
    pricein = []
    priceout = []
    price_in = 1



    ########策略逻辑
    if choice==1:#策略逻辑：指标移动平均上升就买进

        for i in range(0, pdatas.shape[0] - 1):
            if (pdatas.zb[i]>0):

                pdatas.flag[i] = 1
                pdatas.position[i+1] = 1
                date_in = pdatas.index[i]
                price_in = pdatas.asset[i]
                pricein.append([date_in, price_in])
            if (pdatas.zb[i] <=0):

                pdatas.flag[i] = -1
                pdatas.position[i+1] = 0
                date_out = pdatas.index[i]
                price_out = pdatas.asset[i]
                priceout.append([date_out, price_out])
    elif choice == -1:#策略逻辑：指标移动平均下降就买进
        for i in range(0, pdatas.shape[0] - 1):
            if (pdatas.zb[i] < 0):

                pdatas.flag[i] = 1
                pdatas.position[i + 1] = 1
                date_in = pdatas.index[i]
                price_in = pdatas.asset[i]
                pricein.append([date_in, price_in])
            if (pdatas.zb[i] >= 0):
                pdatas.flag[i] = -1
                pdatas.position[i + 1] = 0
                date_out = pdatas.index[i]
                price_out = pdatas.asset[i]
                priceout.append([date_out, price_out])

    p1 = pd.DataFrame(pricein, columns=['datebuy', 'pricebuy'])
    p2 = pd.DataFrame(priceout, columns=['datesell', 'pricesell'])

    transactions = pd.concat([p1, p2], axis=1)
    pdatas['ret'] = pdatas.asset.pct_change(1).fillna(0)
    pdatas['nav'] = (1 + pdatas.ret * pdatas.position).cumprod()
    pdatas['benchmark'] = pdatas.asset / pdatas.asset[0]

    df = pd.DataFrame(pdatas.index.strftime('%Y-%m-%d').str.split('-').tolist(),
                      columns=['year', 'month', 'day'], dtype=int)
    pdatas['ii'] = np.arange(pdatas.shape[0])
    pdatas['da'] = pdatas.index
    pdatas.set_index(pdatas['ii'], inplace=True)
    pdatas = pd.concat([pdatas, df], axis=1)
    pdatas.set_index(pdatas['da'], inplace=True)

    pdatas.to_csv('result.csv')

    stats = performace(transactions, pdatas)

    return stats, transactions, pdatas


def performace(transactions, strategy):

    N = 3
    print(strategy)
    print(strategy.nav)

    # 计算年化收益率
    rety = strategy.nav[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1
    bench_rety = strategy.benchmark[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1

    # 计算夏普比
    Sharp = ((strategy.ret * strategy.position).mean() )/( (strategy.ret * strategy.position).std() * np.sqrt(N))

    # 计算胜率
    VictoryRatio = ((transactions.pricesell - transactions.pricebuy) > 0).mean()

    DD = 1 - strategy.nav / strategy.nav.cummax()
    MDD = max(DD)

    # 作图
    xtick = np.round(np.linspace(0, strategy.shape[0] - 1, 7), 0).astype(int)
    xticklabel = strategy.index[xtick].strftime("%y %b")

    plt.figure(figsize=(9, 4))
    ax1 = plt.axes()
    plt.plot(np.arange(strategy.shape[0]), strategy.benchmark, 'black', label='benchmark', linewidth=2)
    plt.plot(np.arange(strategy.shape[0]), strategy.nav, 'red', label='nav', linewidth=2)
    # plt.plot(np.arange(strategy.shape[0]), strategy.nav / strategy.benchmark, 'orange', label='RS', linewidth=2)
    # plt.plot(np.arange(strategy.shape[0]), 1 , 'grey', label='1', linewidth=1)

    plt.plot(np.arange(strategy.shape[0]), strategy.zb/10 , 'orange', label='zb', linewidth=2)

    # plt.plot(np.arange(strategy.shape[0]), strategy.jtf_roll/50000+1 , 'blue', label='jtf_roll', linewidth=2)
    # lim = [1] * 120
    # plt.plot(lim, "r--")

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
              'bench_rety': bench_rety,
              'WinRate': VictoryRatio,
              'MDD': MDD,
              'maxlossOnce': -maxloss,
              'num': round(strategy.flag.abs().sum() / strategy.shape[0], 1)}




    return result






####################################################
choice=-1#1为指标上升买入策略 -1为指标下降买入策略
zhibiao_result=pd.DataFrame()

#载入数据
assets=pd.read_csv('zhishu.csv')
assets=assets.copy()
assets['date']=pd.to_datetime(assets['date'])
assets.set_index(assets['date'],inplace=True)
del assets['date']


hg_y=pd.read_csv('hg_ji.csv')
hg_y['date']=pd.to_datetime(hg_y['date'])
hg_y.set_index(hg_y['date'],inplace=True)
del hg_y['date']
hg_y=pd.DataFrame(hg_y)

sharp=pd.DataFrame()
# 选取检测对
zhibiao=hg_y['gdp_xx']
asset=assets['hs300']

print(zhibiao)
print(asset)
pdatas=cleanData(asset,zhibiao)
result=Strategy(pdatas)
print(result)


