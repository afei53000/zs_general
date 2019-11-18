**用来一一检测单项宏观指标和单项资产关系。 可导出回测得到的「图形」，和print出「年化收益率、benchmark、夏普比」等回测指标**

需在底部输入宏观指标和资产对应的代码。 如：

>选取检测对<br/>
zhibiao=hg_y['gdp_xx']<br/>
asset=assets['hs300']

对应检测【GDP:不变价:信息传输、软件和信息技术服务业:当季同比】和【沪深300】之间的显著关系。


## 以下是代码对应关系表：

资产

>zx_1	=	综合(中信)	<br/>
zx_2	=	综合Ⅱ(中信)	<br/>
zx_3	=	综合Ⅲ(中信)	<br/>
hs300	=	沪深301	<br/>
zhongzhai10	=	中债-11年期国债财富(总值)指数	<br/>
nh_gyp	=	南华工业品指数	<br/>
nh_ncp	=	南华农产品指数	<br/>
nh_js	=	南华金属指数	<br/>
nh_nh	=	南华能化指数	<br/>
zz500	=	中证500	<br/>
zz1000	=	中证1000	<br/>
sz50	=	上证51	<br/>
hs300_cz	=	沪深301成长	<br/>
hs301_jz	=	沪深301价值	<br/>
zz500_cz	=	中证501成长	<br/>
zz500_jz	=	中证501价值	<br/>
300zq	=	301周期	<br/>
300fz	=	301非周	<br/>


季度数据
>gdp	=	GDP:不变价:当季同比	<br/>
gdp_1	=	GDP:不变价:第一产业:当季同比	<br/>
gdp_2	=	GDP:不变价:第二产业:当季同比	<br/>
gdp_3	=	GDP:不变价:第三产业:当季同比	<br/>
gdp_nlmy	=	GDP:不变价:农林牧渔业:当季同比	<br/>
gdp_gy	=	GDP:不变价:工业:当季同比	<br/>
gdp_zzy	=	GDP:不变价:工业:制造业:当季同比	<br/>
gdp_jzy	=	GDP:不变价:建筑业:当季同比	<br/>
gdp_pfls	=	GDP:不变价:批发和零售业:当季同比	<br/>
gdp_jt	=	GDP:不变价:交通运输、仓储和邮政业:当季同比	<br/>
gdp_zscy	=	GDP:不变价:住宿和餐饮业:当季同比	<br/>
gdp_jr	=	GDP:不变价:金融业:当季同比	<br/>
gdp_fdc	=	GDP:不变价:房地产业:当季同比	<br/>
gdp_xx	=	GDP:不变价:信息传输、软件和信息技术服务业:当季同比	<br/>
gdp_zl	=	GDP:不变价:租赁和商务服务业:当季同比	<br/>
gdp_qt	=	GDP:不变价:其他行业:当季同比	<br/>
jchbye	=	基础货币余额:同比	<br/>
ccl	=	超额存款准备金率(超储率):金融机构	<br/>
syyh_ldxbl	=	商业银行:流动性比例	<br/>
syyh_cdb	=	商业银行:存贷比	<br/>
syyh_jxc	=	商业银行:净息差	<br/>
shrzgm_cl	=	社会融资规模存量:同比	<br/>
syl	=	城镇登记失业率	<br/>
sybxj	=	城镇领取失业保险金人数	<br/>


月度数据
>kqzs	=	克强指数:当月值	<br/>
gyzjz	=	工业增加值:当月同比	<br/>
gyzjz_qgy	=	工业增加值:轻工业:当月同比	<br/>
gyzjz_zgy	=	工业增加值:重工业:当月同比	<br/>
gyzjz_cky	=	工业增加值:采矿业:当月同比	<br/>
gyzjz_zzy	=	工业增加值:制造业:当月同比	<br/>
gyzjz_gy	=	工业增加值:国有及国有控股企业:当月同比	<br/>
gyzjz_jt	=	工业增加值:集体企业:当月同比	<br/>
gyzjz_gfhz	=	工业增加值:股份合作企业:当月同比	<br/>
gyzjz_gf	=	工业增加值:股份制企业:当月同比	<br/>
gyzjz_ws	=	工业增加值:外商及港澳台投资企业:当月同比	<br/>
gyzjz_sy	=	工业增加值:私营企业:当月同比	<br/>
chanliang_ym	=	产量:原煤:当月同比	<br/>
chanliang_jt	=	产量:焦炭:当月同比	<br/>
chanliang_fdl	=	产量:发电量:当月同比	<br/>
chanliang_trq	=	产量:天然气:当月同比	<br/>
chanliang_gc	=	产量:钢材:当月同比	<br/>
chanliang_tc	=	产量:铜材:当月同比	<br/>
chanliang_xny	=	产量:新能源汽车:当月同比	<br/>
chanliang_cd	=	产量:彩电:当月同比	<br/>
chanliang_kt	=	产量:空调:当月同比	<br/>
chanliang_bx	=	产量:家用电冰箱:当月同比	<br/>
chanliang_xyj	=	产量:家用洗衣机:当月同比	<br/>
pmi	=	PMI	<br/>
pmi_sc	=	PMI:生产	<br/>
pmi_xdd	=	PMI:新订单	<br/>
pmi_xckdd	=	PMI:新出口订单	<br/>
pmi_zsdd	=	PMI:在手订单	<br/>
pmi_ccpkc	=	PMI:产成品库存	<br/>
pmi_cgl	=	PMI:采购量	<br/>
pmi_jk	=	PMI:进口	<br/>
pmi_ccjg	=	PMI:出厂价格	<br/>
pmi_ycljg	=	PMI:主要原材料购进价格	<br/>
pmi__yclkc	=	PMI:原材料库存	<br/>
pmi_cyry	=	PMI:从业人员	<br/>
pmi_pssj	=	PMI:供货商配送时间	<br/>
pmi_yq	=	PMI:生产经营活动预期	<br/>
cpi	=	CPI:当月同比	<br/>
cpi_sp	=	CPI:食品:当月同比	<br/>
cpi_fsp	=	CPI:非食品:当月同比	<br/>
cpi_hx	=	CPI:不包括食品和能源(核心CPI):当月同比	<br/>
cpi_xfp	=	CPI:消费品:当月同比	<br/>
cpi_fw	=	CPI:服务:当月同比	<br/>
ppi_gyp	=	PPI:全部工业品:当月同比	<br/>
rpi	=	RPI:当月同比	<br/>
M1	=	M1:同比	<br/>
M2	=	M2:同比	<br/>
gbcs	=	货币乘数	<br/>
gdzc	=	固定资产投资完成额:累计同比	<br/>
gdzc_xz	=	新增固定资产投资完成额:累计同比	<br/>
gdzc_fdc	=	占固定资产投资完成额比重:房地产开发	<br/>
gdzc_1	=	占固定资产投资完成额比重:第一产业	<br/>
gdzc_2	=	占固定资产投资完成额比重:第二产业	<br/>
gdzc_3	=	占固定资产投资完成额比重:第三产业	<br/>
gdzc_zyxm	=	占固定资产投资完成额比重:中央项目	<br/>
gdzc_dfxm	=	占固定资产投资完成额比重:地方项目	<br/>
jckje	=	进出口金额:当月同比	<br/>
ckje	=	出口金额:当月同比	<br/>
yhjshce	=	银行结售汇差额:当月值	<br/>
shxfplsze	=	社会消费品零售总额:当月同比	<br/>
zqfxl_gz	=	债券发行量:记账式国债:当月值	<br/>
shrzgm	=	社会融资规模:当月值	<br/>
jqzs_yj	=	宏观经济景气指数:预警指数	<br/>
jqzs_yz	=	宏观经济景气指数:一致指数	<br/>
jqzs_xx	=	宏观经济景气指数:先行指数	<br/>
jqzs_zh	=	宏观经济景气指数:滞后指数	<br/>



