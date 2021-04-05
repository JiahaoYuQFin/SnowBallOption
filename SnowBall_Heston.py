import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time


def draw_fig(x, y, x_label, title,issave, isshow=True):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 8)

    ax.plot(x, y, color='r', linewidth=3)
    ax.set_title(title)
    ax.grid(True)
    ax_yticks = np.arange(np.min(y)+1, np.max(y)-1, 1)
    ax.set_yticks(ax_yticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Option Payoff/ %')

    # 解决中文字体问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if issave == True:
        plt.savefig(title+'.png')
    if isshow == True:
        plt.show()

def draw_scatter(date, S0, v0, S_Heston, S_BS, npath, issave, isshow = True):
    fig, axis = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(15, 8)
    ax1 = axis[0]
    ax2 = axis[1]

    for i in range(npath // 2):
        ax1.scatter(date, S_Heston[:, i], marker='.', color='black', alpha=0.05)
        ax2.scatter(date, S_BS[:, i], marker='.', color='black', alpha=0.05)
    title1 = 'Heston multipath Prices with S0='+str(S0)+' sqrt(v0)='+str(np.sqrt(v0))
    title2 = 'GBM multipath Prices with S0='+str(S0)+' sqrt(v0)='+str(np.sqrt(v0))
    ax1.set_title(title1)
    ax1.grid(True)
    ax2.set_title(title2)
    ax2.grid(True)
    if issave == True:
        plt.savefig(title1+'.png')
    if isshow == True:
        plt.show()

def draw_vt_scatter(date, S0, v, npath, issave, isshow = True):
    sqrt_v = np.sqrt(v)
    fig, axis = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(15, 8)
    ax1 = axis[0]
    ax2 = axis[1]

    ax1.plot(date, sqrt_v[:, 0], color='red', linewidth=2)
    for i in range(npath // 2):
        ax2.scatter(date, sqrt_v[:, i], marker='.', color='black', alpha=0.05)
    title1 = 'Heston Model -- Parameters sqrt(v) with S0='+str(S0)+' sqrt(v0)='+str(sqrt_v[0,0])
    title2 = 'sqrt(v) under multipath with S0='+str(S0)+' sqrt(v0)='+str(sqrt_v[0,0])
    ax1.set_title(title1)
    ax1.grid(True)
    ax2.set_title(title2)
    ax2.grid(True)
    if issave == True:
        plt.savefig(title1+'.png')
    if isshow == True:
        plt.show()

def SnowballOption(
        StockPrice,
        y,  # 期初年份
        m,  # 期初月份
        d,  # 期初日期
        T,  # 合约剩余期限/年
        B,  # 投入本金
        r_peryear,  # 票息率
        r,  # 利率
        S0,  # 给定股价
        K1,  # 敲出价
        K2,  # 敲入价
        npath,  # MC模拟路径数,包括对偶路径
        date,
        d_start,
        nstep
):
    syb_up = np.zeros((nstep+1, npath // 2 * 2)) # 记录向上敲出时候的收益
    syb_up_anti = np.ones((1, npath // 2 * 2)) # 向上敲出01值的反转
    syb_down_K2 = np.copy(syb_up_anti)      # 向下敲出的01值
    syb_down_S0 = np.zeros((1, npath // 2 * 2))     # 向下敲出时候期末股价低于S0
    payoff_discounted_npath = np.zeros((1, npath // 2 * 2))

    for t in range(1, nstep):
        if date[t].day == d:
            syb_up[t, :] = syb_up[t, :] + (StockPrice[t, :] >= K1)
            temp = (date[t].year - d_start.year) * 12 + (date[t].month - d_start.month)
            syb_up[t, :] = syb_up[t, :] * B * r_peryear * temp * 1 / 12 * np.exp(-r * temp / 12)
            syb_up_anti = syb_up_anti * (StockPrice[t, :] < K1)
        syb_down_K2 = syb_down_K2 * (StockPrice[t, :] >= K2)        # 此处其实是anti，后面会反转

    syb_up_nonzero_id = np.where((syb_up != 0).any(axis=0), (syb_up != 0).argmax(axis=0), 0)
    syb_down_S0 = syb_down_S0 + (StockPrice[-1, :] < S0)
    syb_down_K2_anti = syb_down_K2 * B * r_peryear * np.exp(-r*T)

    # 小于K2信号中0、1互换
    for col in range(npath//2*2):
        if syb_down_K2[0, col] == 1:
            syb_down_K2[0, col] = 0
        else:
            syb_down_K2[0, col] = 1
        payoff_discounted_npath[0, col] = syb_up[syb_up_nonzero_id[col], col]

    # 计算payoff
    payoff_discounted_npath = payoff_discounted_npath + \
                              syb_up_anti*(syb_down_K2_anti +
                               syb_down_K2 * syb_down_S0 * (StockPrice[-1, :] / S0 - 1) * B * np.exp(-r * T))

    # for ipath in range(npath//2*2):
    #     syb_up_ipath = syb_up[:, ipath]
    #
    #     if len(syb_up_ipath[np.nonzero(syb_up_ipath)]) > 0:
    #         payoff_discounted_npath[0, ipath] = syb_up_ipath[np.nonzero(syb_up_ipath)][0]
    #     elif syb_down_K2[0, ipath] == 0:
    #         payoff_discounted_npath[0, ipath] = syb_down_K2_anti[0, ipath]
    #     else:
    #         if isinstance(S0, (int, float)):
    #             payoff_discounted_npath[0, ipath] = (StockPrice[-1, ipath] / S0 - 1) * B * np.exp(-r*T) \
    #                                             * syb_down_S0[0, ipath] * syb_down_K2[0, ipath]
    #         else:
    #             payoff_discounted_npath[0, ipath] = (StockPrice[-1, ipath] / S0[ipath] - 1) * B * np.exp(-r * T) \
    #                                                 * syb_down_S0[0, ipath] * syb_down_K2[0, ipath]

    # print(payoff_discounted_npath)
    # print(np.mean(payoff_discounted_npath))
    # print('\n')
    # print(payoff_discounted_npath_best)
    # print(np.mean(payoff_discounted_npath_best))

    return np.mean(payoff_discounted_npath)


def StockModel(
        d_start,    # 期初年月日
        d_end,      # 期末年月日
        T,          # 合约期限/年
        r,          # 利率
        S0,         # 给定股价
        npath,      # MC模拟路径数,包括对偶路径
        kappa,      # Heston模型dt参数
        theta,      # Heston模型dt参数
        sigma,      # Heston模型dWt参数
        rho,        # Heston模型2个维纳过程相关系数
        v0,         # Heston模型初始方差
        model_select, # Heston or GBM
        isdraw      # 是否绘股价路径图
):


    date = pd.date_range(d_start, d_end, freq='1D')
    nstep = (d_end - d_start).days
    dt = T/nstep


    CovWeiner = np.array([[1., rho], [rho, 1.]])
    L = np.linalg.cholesky(CovWeiner)
    S_Heston = S0*np.ones((nstep+1, npath//2*2), dtype='float32')
    S_BS = S0*np.ones((nstep+1, npath//2*2), dtype='float32')
    v = v0*np.ones((nstep+1, npath//2*2), dtype='float32')

    ## 生成股票价格路径及对偶路径
    for k in range(0, nstep):
        WhiteNoise = np.random.normal(loc=0,scale=1,size=[2, npath//2]) # 生成二维独立随机变量
        WhiteNoise_iid = np.c_[WhiteNoise, -WhiteNoise] # 生成对偶路径，组成整个npath
        WhiteNoise_correlated = L @ WhiteNoise_iid # 生成二维相关随机变量

        S_Heston[k + 1, :] = S_Heston[k, :] * np.exp((r - 1/2 * v[k,:])*dt + np.sqrt(v[k,:]*dt)*WhiteNoise_correlated[0,:])
        v[k+1, :] = v[k, :] * np.exp((kappa*(theta-v[k,:])-1/2 * (sigma**2))*dt/v[k,:]
                                     + sigma * np.sqrt(dt/v[k,:]) * WhiteNoise_correlated[1,:])
        S_BS[k+1, :] = S_BS[k, :] * np.exp((r - 1/2 * v0)*dt + np.sqrt(v0*dt)*WhiteNoise_correlated[0,:])

    # 画散点路径图
    if isdraw == True:
        # 画Heston模型和GBM模型股价散点图
        draw_scatter(date, S0, v0, S_Heston, S_BS, npath, issave=True)
        # 画Heston模型的参数sqrt(v)的时间变化
        draw_vt_scatter(date, S0, v, npath, issave=True)

    if model_select == 'Heston':
        return S_Heston, date, nstep
    elif model_select == 'GBM':
        return S_BS, date, nstep
    else:
        print('model_select error!')
        return 'Error'



def main():
    #global y, m, d,T , B, r_peryear, r, K1, K2, N
    y = 2021
    m = 2
    d = 16
    T = 1           # 合约期限/年
    B = 100
    r_peryear = 0.18
    r = 0.05
    S0=90           # 给定股价
    K1=100          # 敲出价
    K2=80           # 敲入价
    npath=10000     # MC模拟路径数,包括对偶路径
    kappa=2         # Heston模型dt参数
    theta=0.04      # Heston模型dt参数
    sigma=0.1       # Heston模型dWt参数
    rho=-0.7        # Heston模型2个维纳过程相关系数
    v0=0.02         # Heston模型初始方差
    isdraw = False   # 是否绘股价路径图
    model = ['Heston', 'GBM']
    model_select = model[0]

    opt_s0 = []
    opt_v0 = []
    opt_day = []
    s0_range = np.linspace(70, 110, 100)
    v0_range = np.linspace(0.01, 0.09, 100)
    d_start = datetime.datetime(y, m, d)
    d_end = datetime.datetime(y+T, m, d)
    day0_range = pd.date_range(d_start, d_end, freq='1D')

    # # 股票路径生成
    # stock_params = [d_start, d_end, T, r, S0, npath, kappa, theta, sigma, rho, v0, model_select, isdraw]
    # stockprice_npath, date, nstep = StockModel(*stock_params)
    #
    # # 判断衍生品敲入敲出条件，并计算价格
    # judge_params = [stockprice_npath, y, m, d, T, B, r_peryear, r, S0, K1, K2, npath, date, d_start, nstep]
    # SnowballOption(*judge_params)




    for s_start in s0_range:
        # 股票路径生成
        stock_params = [d_start, d_end, T, r, s_start, npath, kappa, theta, sigma, rho, v0, model_select, isdraw]
        stockprice_npath, date, nstep = StockModel(*stock_params)

        # 判断衍生品敲入敲出条件，并计算价格
        judge_params = [stockprice_npath, y, m, d, T, B, r_peryear, r, S0, K1, K2, npath, date, d_start, nstep]
        opt_price = SnowballOption(*judge_params)

        opt_s0.append(opt_price)

    title_delta = model_select+'模型的Delta 模拟路径npath='+str(npath)+' sqrt(v0)='+str(np.around(np.sqrt(v0), 2))
    draw_fig(s0_range, opt_s0, x_label='Stock Price/ $', title=title_delta, issave=True)

    for v_start in v0_range:
        # 股票路径生成
        stock_params = [d_start, d_end, T, r, S0, npath, kappa, theta, sigma, rho, v_start, model_select, isdraw]
        stockprice_npath, date, nstep = StockModel(*stock_params)

        # 判断衍生品敲入敲出条件，并计算价格
        judge_params = [stockprice_npath, y, m, d, T, B, r_peryear, r, S0, K1, K2, npath, date, d_start, nstep]
        opt_price = SnowballOption(*judge_params)
        opt_v0.append(opt_price)

    title_vega = model_select+'模型的Vega 模拟路径npath='+str(npath)
    draw_fig(v0_range*100, opt_v0, x_label='v0/ % (初始方差)', title=title_vega, issave=True)

    # Theta 计算衍生品随时间变动
    # 股票路径生成
    stock_params = [d_start, d_end, T, r, S0, npath, kappa, theta, sigma, rho, v0, model_select, isdraw]
    stockprice_npath_ori, date_ori, nstep = StockModel(*stock_params)
    for t in range(len(date_ori)-1):
        stockprice_npath = stockprice_npath_ori[t:, :]
        s_start = stockprice_npath_ori[t, :]
        date = date_ori[t:]
        day0 = date_ori[t]
        nstep = (d_end-day0).days
        T_varying = nstep / 365
        # 判断衍生品敲入敲出条件，并计算价格
        judge_params = [stockprice_npath, y, m, d, T_varying, B, r_peryear, r, s_start, K1, K2, npath, date, day0, nstep]
        opt_price = SnowballOption(*judge_params)
        opt_day.append(opt_price)

    title_theta = model_select+'模型的Theta 模拟路径npath='+str(npath)+' sqrt(v0)='+str(np.around(np.sqrt(v0), 2))
    draw_fig(day0_range[:-1], opt_day, x_label='Date', title=title_theta, issave=True)


if __name__ == '__main__':
    main()