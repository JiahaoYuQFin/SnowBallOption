import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 对每一行股价求相应期权pay
# def DailyOptionPrice(Stock, StockPrice, CurrentDay, InitialDay, d_start):
#
#     if CurrentDay.day == InitialDay and StockPrice >= K1 :
#         Stock['syb_up'] = 1
#         temp = CurrentDay.month - d_start.month
#         Stock['OptPayoff']= B * r_peryear * (temp) * 1 / 12 * np.exp(-r * temp / 12)
#     elif StockPrice < K2:
#         Stock['syb_down'] = 1
#     return Stock

def draw_fig_payoff(df, issave, isshow=True):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(15, 8)
    ax2 = ax1.twinx()

    p1, = ax1.plot(df.index, df['StockPrice'], color='r', linewidth=2)
    p2, = ax2.plot(df.index, df['OptPayoff'], color='steelblue', linewidth=4)
    title = 'Price and Payoff from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    ax1.set_title(title)
    ax1.legend([p1, p2], ['Stock', 'Option'])
    ax1.grid(True)
    ax2.set_xlim(df.index[0], df.index[-1])
    ax1.set_ylabel('Stock Price/ $')
    ax2.set_ylabel('Option Payoff/ %')

    # f2 = ax2.fill_between(df.index, df['MaxDD_capital_control'], 0, facecolor='r', alpha=0.5)
    # f3 = ax2.fill_between(df.index, df['MaxDD_capital'], 0, facecolor='steelblue', alpha=0.5)
    # title2 = 'Drawdown from %s to %s' % (df.index.astype(str)[0], df.index.astype(str)[-1])
    # ax2.legend([f2, f3], ['MaxDD_capital_control', 'MaxDD_capital'], loc=3)
    # ax2.set_title(title2)
    # ax2.grid(True)
    # ax2.set_xlim(df.index[0], df.index[-1])

    if issave == True:
        plt.savefig(title+'.png')
    if isshow == True:
        plt.show()

def draw_fig(x, y, x_label, title,issave, isshow=True):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 8)

    p1, = ax.plot(x, y, color='r', linewidth=3)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Option Payoff/ %')

    # 解决中文字体问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if issave == True:
        plt.savefig(title+'.png')
    if isshow == True:
        plt.show()


# 计算单次模拟后对偶路径平均的平均期权的payoff
def payoff_calculate(d_start, d_end, td, S0, sigma, rdm):
    StockPrice_df = pd.DataFrame(index=pd.date_range(d_start, d_end, freq='1D'), columns=
    ['CumGrowth_S', 'StockPrice', 'OptPayoff', 'syb_up', 'syb_down'])
    dt = np.diff(td)
    StockPrice_df['CumGrowth_S'] = np.hstack((S0, np.exp((r - 0.5 * (sigma ** 2)) * dt + sigma * np.sqrt(dt) * rdm)))
    StockPrice_df['StockPrice'] = StockPrice_df['CumGrowth_S'].cumprod()
    StockPrice_df['OptPayoff'] = 0
    syb_up=0
    syb_down=0
    payoff_discounted=0
    # StockPrice_df = StockPrice_df.apply(lambda x: DailyOptionPrice(x, x['StockPrice'], x.name, d, d_start), axis=1)

    # if np.isnan(StockPrice_df['syb_up'].all()):
    #     syb_up = 0
    # else:
    #     syb_up = 1
    #
    # if np.isnan(StockPrice_df['syb_down'].all()):
    #     syb_down = 0
    # else:
    #     syb_down = 1
    #
    # payoff=StockPrice_df['OptPayoff'].dropna()
    # try:
    #     payoff_discounted = payoff[1]
    # except Exception:
    #     payoff_discounted = 0
    for eachday in StockPrice_df.index[1:]:

        if eachday.day == d and StockPrice_df.loc[eachday,'StockPrice'] >= K1:
            syb_up = 1
            temp = (eachday.year - d_start.year)*12+(eachday.month - d_start.month)
            payoff_discounted = B * r_peryear * (temp) * 1 / 12 * np.exp(-r * temp / 12)
            StockPrice_df.loc[eachday, 'OptPayoff'] = payoff_discounted

            break
        elif StockPrice_df.loc[eachday,'StockPrice'] < K2:
            syb_down = 1

    if syb_up == 0 and syb_down == 0:
        payoff_discounted = B * r_peryear * np.exp(-r)
        StockPrice_df.loc[StockPrice_df.index[-1],'OptPayoff'] = payoff_discounted
    elif syb_up == 0 and syb_down == 1 and StockPrice_df['StockPrice'][-1] < S0:
        payoff_discounted = (StockPrice_df['StockPrice'][-1] / S0 - 1) * B * np.exp(-r)
        StockPrice_df.loc[StockPrice_df.index[-1],'OptPayoff'] = payoff_discounted
    elif syb_up == 0 and syb_down == 1 and StockPrice_df['StockPrice'][-1] >= S0:
        payoff_discounted = 0
        StockPrice_df.loc[StockPrice_df.index[-1],'OptPayoff'] = payoff_discounted

    #draw_fig_payoff(df=StockPrice_df,issave=False)

    return payoff_discounted


##最好在每个月28号前做交易
## 给定票息率得到期权价格


def snowball_12_opt(
        # y,  # 期初年份
        # m,  # 期初月份
        # d,  # 期初日期
        # B,  # 投入本金
        # r_peryear,  # 票息率
        # r,  # 利率
        sigma,  # 波动率
        S0  # 给定股价
        # K1,  # 敲出价
        # K2,  # 敲入价
        # N  # 模拟路径数
):


    d_start = datetime.datetime(y, m, d)
    d_end = datetime.datetime(y + 1, m, d)
    n = (d_end - d_start).days
    ## 生成路径及折现收益
    #global payoff_discounted,k

    payoff_discounted = np.zeros((N, 1))
    payoff_anti = np.zeros((N, 1))
    for k in range(N):
        rdm = np.random.randn(n)
        td = np.linspace(0, 1, n + 1)


        ## 对偶路径
        payoff_anti[k] = payoff_calculate(d_start, d_end, td, S0, sigma, -rdm)
        # 生成路径
        payoff_discounted[k] = payoff_calculate(d_start, d_end, td, S0, sigma, rdm)

        # SA = np.zeros((n + 1, 1))
        # SA[0] = S0
        # syb_up = 0
        # syb_down = 0
        # for i in range(n):
        #     date = d_start + datetime.timedelta(days=i + 1)
        #     dt = td[i + 1] - td[i]
        #     SA[i + 1] = SA[i] * np.exp((r - 0.5 * (sigma ** 2)) * dt + sigma * np.sqrt(dt) * (-1 * rdm[i]))
        #     if date.day == d and SA[i + 1] >= K1:
        #         syb_up = 1
        #         temp = date.month - d_start.month
        #         payoff_anti[k] = B * r_peryear * (temp) * 1 / 12 * np.exp(-r * temp / 12)
        #         break
        #     elif SA[i + 1] < K2:
        #         syb_down = 1
        # if syb_up == 0 and syb_down == 0:
        #     payoff_anti[k] = B * r_peryear * np.exp(-r)
        # elif syb_up == 0 and syb_down == 1 and SA[n] < S0:
        #     payoff_anti[k] = (SA[n] / S0 - 1) * B * np.exp(-r)
        # elif syb_up == 0 and syb_down == 1 and SA[n] >= S0:
        #     payoff_anti[k] = 0
    ## MC
    return 0.5 * np.mean(payoff_discounted + payoff_anti)/B*100


def main():
    opt_price = []
    opt_sigma = []
    global y, m, d, B, r_peryear, r, K1, K2, N
    y = 2021
    m = 1
    d = 17
    B = 100
    r_peryear = 0.18
    r = 0.2
    K1 = 100
    K2 = 80
    N = 10000
    stock_range = np.linspace(80,100,30)
    sigma_range = np.linspace(0.1,0.3,30)

    # snowball_12_opt(sigma=0.2, S0=90)
    for stock in stock_range:
        # s0.append(
        #     snowball_12_opt(y=2021, m=1, d=16, B=100000, r_peryear=0.18,
        #                     r=0.2, sigma=0.2, S0=stock, K1=100, K2=80,
        #                     N=1))
        opt_price.append(snowball_12_opt(sigma=0.2,S0=stock))

    draw_fig(stock_range, opt_price, x_label='Stock Price/ $',title='Delta 模拟次数N='+str(N),issave=True)

    # opt_delta = np.diff(opt_price)

    for sigma in sigma_range:
        opt_sigma.append(snowball_12_opt(sigma=sigma,S0=90))

    draw_fig(sigma_range*100, opt_sigma, x_label='sigma/ %', title='Vega 模拟次数N='+str(N), issave=True)

main()