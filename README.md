**用来一一检测单项宏观指标和单项资产关系。 可导出回测得到的「图形」，和print出「年化收益率、benchmark、夏普比」等回测指标**

需在底部输入宏观指标和资产对应的代码。 如：

>选取检测对
zhibiao=hg_y['gdp_xx']
asset=assets['hs300']

对应检测【GDP:不变价:信息传输、软件和信息技术服务业:当季同比】和【沪深300】之间的显著关系。


## 以下是代码对应关系表：

资产

>zx_1	综合(中信)
zx_2	综合Ⅱ(中信)
zx_3	综合Ⅲ(中信)
hs300	沪深300
zhongzhai10	中债-10年期国债财富(总值)指数
nh_gyp	南华工业品指数
nh_ncp	南华农产品指数
nh_js	南华金属指数
nh_nh	南华能化指数
zz500	中证500
zz1000	中证1000
sz50	上证50
hs300_cz	沪深300成长
hs301_jz	沪深300价值
zz500_cz	中证500成长
zz500_jz	中证500价值
300zq	300周期
300fz	300非周

季度数据
>gdp	GDP:不变价:当季同比
gdp_1	GDP:不变价:第一产业:当季同比
gdp_2	GDP:不变价:第二产业:当季同比
gdp_3	GDP:不变价:第三产业:当季同比
gdp_nlmy	GDP:不变价:农林牧渔业:当季同比
gdp_gy	GDP:不变价:工业:当季同比
gdp_zzy	GDP:不变价:工业:制造业:当季同比
gdp_jzy	GDP:不变价:建筑业:当季同比
gdp_pfls	GDP:不变价:批发和零售业:当季同比
gdp_jt	GDP:不变价:交通运输、仓储和邮政业:当季同比
gdp_zscy	GDP:不变价:住宿和餐饮业:当季同比
gdp_jr	GDP:不变价:金融业:当季同比
gdp_fdc	GDP:不变价:房地产业:当季同比
gdp_xx	GDP:不变价:信息传输、软件和信息技术服务业:当季同比
gdp_zl	GDP:不变价:租赁和商务服务业:当季同比
gdp_qt	GDP:不变价:其他行业:当季同比
jchbye	基础货币余额:同比
ccl	超额存款准备金率(超储率):金融机构
syyh_ldxbl	商业银行:流动性比例
syyh_cdb	商业银行:存贷比
syyh_jxc	商业银行:净息差
shrzgm_cl	社会融资规模存量:同比
syl	城镇登记失业率
sybxj	城镇领取失业保险金人数

月度数据
>kqzs	克强指数:当月值
gyzjz	工业增加值:当月同比
gyzjz_qgy	工业增加值:轻工业:当月同比
gyzjz_zgy	工业增加值:重工业:当月同比
gyzjz_cky	工业增加值:采矿业:当月同比
gyzjz_zzy	工业增加值:制造业:当月同比
gyzjz_gy	工业增加值:国有及国有控股企业:当月同比
gyzjz_jt	工业增加值:集体企业:当月同比
gyzjz_gfhz	工业增加值:股份合作企业:当月同比
gyzjz_gf	工业增加值:股份制企业:当月同比
gyzjz_ws	工业增加值:外商及港澳台投资企业:当月同比
gyzjz_sy	工业增加值:私营企业:当月同比
chanliang_ym	产量:原煤:当月同比
chanliang_jt	产量:焦炭:当月同比
chanliang_fdl	产量:发电量:当月同比
chanliang_trq	产量:天然气:当月同比
chanliang_gc	产量:钢材:当月同比
chanliang_tc	产量:铜材:当月同比
chanliang_xny	产量:新能源汽车:当月同比
chanliang_cd	产量:彩电:当月同比
chanliang_kt	产量:空调:当月同比
chanliang_bx	产量:家用电冰箱:当月同比
chanliang_xyj	产量:家用洗衣机:当月同比
pmi	PMI
pmi_sc	PMI:生产
pmi_xdd	PMI:新订单
pmi_xckdd	PMI:新出口订单
pmi_zsdd	PMI:在手订单
pmi_ccpkc	PMI:产成品库存
pmi_cgl	PMI:采购量
pmi_jk	PMI:进口
pmi_ccjg	PMI:出厂价格
pmi_ycljg	PMI:主要原材料购进价格
pmi__yclkc	PMI:原材料库存
pmi_cyry	PMI:从业人员
pmi_pssj	PMI:供货商配送时间
pmi_yq	PMI:生产经营活动预期
cpi	CPI:当月同比
cpi_sp	CPI:食品:当月同比
cpi_fsp	CPI:非食品:当月同比
cpi_hx	CPI:不包括食品和能源(核心CPI):当月同比
cpi_xfp	CPI:消费品:当月同比
cpi_fw	CPI:服务:当月同比
ppi_gyp	PPI:全部工业品:当月同比
rpi	RPI:当月同比
M1	M1:同比
M2	M2:同比
gbcs	货币乘数
gdzc	固定资产投资完成额:累计同比
gdzc_xz	新增固定资产投资完成额:累计同比
gdzc_fdc	占固定资产投资完成额比重:房地产开发
gdzc_1	占固定资产投资完成额比重:第一产业
gdzc_2	占固定资产投资完成额比重:第二产业
gdzc_3	占固定资产投资完成额比重:第三产业
gdzc_zyxm	占固定资产投资完成额比重:中央项目
gdzc_dfxm	占固定资产投资完成额比重:地方项目
jckje	进出口金额:当月同比
ckje	出口金额:当月同比
yhjshce	银行结售汇差额:当月值
shxfplsze	社会消费品零售总额:当月同比
zqfxl_gz	债券发行量:记账式国债:当月值
shrzgm	社会融资规模:当月值
jqzs_yj	宏观经济景气指数:预警指数
jqzs_yz	宏观经济景气指数:一致指数
jqzs_xx	宏观经济景气指数:先行指数
jqzs_zh	宏观经济景气指数:滞后指数
