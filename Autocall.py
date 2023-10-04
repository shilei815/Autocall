# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:31:26 2023

@author: Lei
"""
import numpy as np
import datetime as dt
import pandas as pd
from  datetime import datetime
from dateutil import parser 
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

def get_trade_date_lst():

    # 获取最新交易日历

    try:

        df_tradedays = pd.read_excel('trade_days.xlsx')

    except:

        path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))

        df_tradedays = pd.read_excel(path+'\\trade_days.xlsx')

    trade_date_lst= list(df_tradedays.iloc[:,0])

    return trade_date_lst

    
def get_delta_trade_days(begin, end):

    ''' 计算交易日间隔

    begin/end:格式是yyyy-mm-dd。

    '''

    trade_date_lst = get_trade_date_lst()

    if begin in trade_date_lst and end in trade_date_lst:
        return trade_date_lst.index(end) - trade_date_lst.index(begin)
    else:
        if not begin in trade_date_lst:
            for i in range(20):
                begin_new = dt.datetime.strftime(dt.datetime.strptime(begin, "%Y-%m-%d") - dt.timedelta(days=i),
                                                 "%Y-%m-%d")  # 如果begin是假期，减去一天直到是交易日。
                if begin_new in trade_date_lst:
                    begin = begin_new
                    break
        if not end in trade_date_lst:
            for i in range(20):
                end_new = dt.datetime.strftime(dt.datetime.strptime(end, "%Y-%m-%d") - dt.timedelta(days=i),
                                               "%Y-%m-%d")  # 如果end是假期，减去一天直到是交易日。
                if end_new in trade_date_lst:
                    end = end_new
                    break
        return trade_date_lst.index(end) - trade_date_lst.index(begin)

def GetActualDaysBetweenTwoDate(begin,end):
    ''' 获取实际天数间隔
    begin/end:格式是yyyy-mm-dd。
    '''
    begin1 = dt.datetime.strptime(begin,'%Y-%m-%d')
    end1 = dt.datetime.strptime(end,'%Y-%m-%d')
    return (end1-begin1).days




class SnowBall:
    
    def __init__(self,parm_dict):
        self.s0 = parm_dict.get('s0',0)                             #期初资产价格
        self.s0_now = parm_dict.get('s0_now',0)                     #目前的标的资产价格
        self.deposit=parm_dict.get('deposit',0)
        self.nominal = parm_dict.get('nominal',0)                   #名义本金
        self.knockin = parm_dict.get('knockin',0)                   #敲入比例
        self.knockout = parm_dict.get('knockout')                   #敲出比例
        self.knockout_Coupon = parm_dict.get('knockout_Coupon',0)   #敲出收益率
        self.knockout_month=parm_dict.get('knockout_month',0)       #封闭月
        self.r = parm_dict.get('r',0)                               #无风险收益率
        self.sigma = parm_dict.get('sigma',0)                       #标的波动率（年化）
        self.pay_Coupon=parm_dict.get('pay_Coupon',0)
        self.numpath = parm_dict.get('numpath',0)                   #MC模拟次数
        self.q = parm_dict.get('q',0)                               #期货贴水率（年化）
        self.T_trading = parm_dict.get('T_trading',240)             #一年的交易日天数
        self.term = parm_dict.get('term',1)                         #期权的期限(年)
        self.StepLength = parm_dict.get('StepLength',21)            #观察日间隔      
        self.val_date = parm_dict.get('val_date')                   #实际估值日期
        self.begin_date = parm_dict.get('begin_date')               #期权开始日期
        self.KnockOutDate = parm_dict.get('KnockOutDate',"-")       #敲出观察日序列
        self.path = parm_dict.get('path')

    def AutoCall_virtual_pv(self):
        '''理论上计算雪球期权的pv'''

        '''为加快运行速度，将类内属性赋值给局部变量'''
        r = self.r
        q = self.q
        sigma = self.sigma
        knockout_Coupon = self.knockout_Coupon
        numpath = self.numpath
        ResidualYears = self.term
        nominal = self.nominal
        acc_income = 0  # 初始化累计收益
        knock_in_price = round(self.knockin * self.s0, 2)  # 敲入价格
        knock_out_price = round(self.knockout * self.s0, 2)  # 敲出价格
        s_next = np.ones(numpath) * self.s0_now  # 期初价格向量
        T_trading = self.T_trading
        Residualdays = int(self.term * self.T_trading)
        StepLength=self.StepLength
        KnockOutTradeStep = list(range(StepLength, Residualdays+1, StepLength))
        knock_in_keeper = np.zeros(numpath)  #标记敲出次数
        knock_out_keeper = np.zeros(numpath)  # 用于记录敲出路径的索引，初始值设为0
        acc_knock_out_times = 0  # 累计敲出笔数
        begin_date=self.begin_date
        term=self.term
        knockout_month=self.knockout_month
        lock_date = (datetime.strptime(begin_date, '%Y-%m-%d') + relativedelta(months=knockout_month)).strftime('%Y-%m-%d')
        t_trade_days = get_delta_trade_days(begin_date, lock_date)

        for i in range(1, Residualdays + 1):
            z = np.random.standard_normal(numpath)
            s_next = s_next * np.exp((r - q - 0.5 * np.square(sigma)) * ResidualYears / Residualdays + sigma * np.sqrt(
                ResidualYears / Residualdays) * z)

            if i in range(1,t_trade_days+1):
                lock_state=0
            else:
                lock_state=1

            # 如果i属于敲出观察日，则需要判断是否敲出，计算模拟数据中有多少笔需要敲出
            if i in KnockOutTradeStep and lock_state==1:
                knock_out_keeper += (s_next >= knock_out_price)
                current_knock_out = np.count_nonzero(knock_out_keeper != 0) - acc_knock_out_times
                acc_knock_out_times += current_knock_out
                # 计算本步敲出收益
                amount_total = nominal * (i / T_trading) * (knockout_Coupon) * current_knock_out

                # print('计息公式：',amount_total+cost_total,nominal,KnockOutActualStep[KnockOutTradeStep.index(i)],knockout_Coupon+pay_Coupon,current_knock_out)
                discount = np.exp(-r * i / T_trading)
                acc_income += amount_total * discount
            # 敲入统计
            knock_in_keeper += (s_next < knock_in_price)

        # 处理未敲出（knock_out_keeper为0的位置）且未敲入（knock_in_keeper为0的位置）
        lst1 = s_next[(knock_out_keeper == 0) & (knock_in_keeper == 0)]
        if len(lst1) != 0:  # 计算有几笔属于未敲出且未敲入
            abs_amount = nominal * term * (knockout_Coupon) * len(lst1)
            discount = np.exp(-r * ResidualYears)
            acc_income += abs_amount * discount

        # 处理未敲出（knock_out_keeper为0的位置）且已敲入（knock_in_keeper不为0的位置）
        lst2 = s_next[(knock_out_keeper == 0) & (knock_in_keeper != 0)]
        if len(lst2) != 0:  # 计算有几笔属于未敲出且已敲入
            abs_amount = np.sum(nominal * (-np.maximum(0, 1 - lst2 / self.s0)))
            acc_income += abs_amount * discount
        return acc_income / numpath
        

    
    
    def AutoCall_true_pv(self):
        '''计算实际期权的pv'''
        '''计算标准雪球的pv'''
        '''参数获取'''
        s0 = self.s0
        s0_now = self.s0_now
        nominal = self.nominal
        deposit = self.deposit
        begin_date = self.begin_date
        val_date = self.val_date
        KnockOutDate = self.KnockOutDate
        knockin = self.knockin
        knockout = self.knockout
        pay_Coupon = self.pay_Coupon
        r = self.r
        q = self.q
        sigma = self.sigma
        numpath = self.numpath
        knockout_Coupon = self.knockout_Coupon
        knockout_month = self.knockout_month
        term = self.term
        knock_in_price = knockin * s0
        knock_out_price = knockout * s0
        lock_date = (datetime.strptime(begin_date, '%Y-%m-%d') + relativedelta(months=knockout_month)).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(begin_date, '%Y-%m-%d') + relativedelta(years=term)).strftime('%Y-%m-%d')


        '''若val_date超出封闭期，则需要判断val_date时刻的s0_now是否敲出'''

        knock_out_status = 0  # 记录val_date时刻的s0_now是否敲出
        if type(KnockOutDate) == str:
            KnockOutDate_lst = KnockOutDate.split(',')
            KnockOutDate_lst.sort(reverse=False)  # 防止敲出观察日不按顺序排列
        else:
            KnockOutDate_lst = KnockOutDate


        if GetActualDaysBetweenTwoDate(lock_date,val_date) > 0:
            if val_date in KnockOutDate_lst:
                if s0_now >= knock_out_price:
                    knock_out_status = 1


        if knock_out_status == 1:

            pv = s0_now
        else:

            '''计算日期数据'''

            # 计算KnockOutTradeStep

            # 列表格式，寻找val_date之后，每个敲出观察日与val_date的交易日间隔天数

            KnockOutTradeStep = []  # 寻找val_date之后,每个交易日(过滤封闭期）与val_date的交易日间隔

            for date in KnockOutDate_lst:
                temp = GetActualDaysBetweenTwoDate(lock_date, date)
                if temp > 0:  # 过滤封闭期之前的敲出日期
                    trade_days_between = get_delta_trade_days(val_date, date)
                    if trade_days_between >= 0:  # 过滤val_date之前的敲出日期
                        KnockOutTradeStep.append(trade_days_between)

            # 计算KnockOutActualStep

            # 列表格式，寻找封闭期后到val_date之间，每个敲出观察日与begin_date的实际间隔天数

            KnockOutActualStep = []

            for date in KnockOutDate_lst:

                temp = GetActualDaysBetweenTwoDate(lock_date, date)

                if temp > 0:  # 过滤封闭期之前的敲出日期

                    actual_days_between = GetActualDaysBetweenTwoDate(val_date, date)

                    if actual_days_between >= 0:  # 过滤val_date之前的敲出日期

                        KnockOutActualStep.append(actual_days_between)

            # 长度修正：需要和KnockOutTradeStep保持一致。

            KnockOutActualStep = KnockOutActualStep[-len(KnockOutTradeStep):]



            '''向量化进行蒙特卡洛模拟'''

            Residualdays = get_delta_trade_days(val_date, end_date)  # 剩余交易日天数

            Days = GetActualDaysBetweenTwoDate(val_date, end_date)  # 剩余实际天数

            ResidualYears = Days / 365  # 计算剩余多少年

            n = 1 * Residualdays  # MC模拟的步数

            delta_t = ResidualYears/n

            knock_out_keeper = np.zeros(numpath)  # 用于记录敲出路径的索引，初始值设为0

            acc_knock_out_times = 0  # 累计敲出笔数


            knock_in_keeper = np.zeros(numpath)  # 用于记录敲入路径的索引。


            s_next = np.ones(numpath) * s0_now  # 期初价格向量

            acc_income = 0  # 累计收益



            for i in range(1, n + 1):  # n+1

                # 第i次指的是MC模拟步数，也是距离val_date的交易日间隔

                z = np.random.standard_normal(numpath)  # 向量化生成随机数

                s_next = s_next * np.exp((r - q - 0.5 * np.square(sigma)) * delta_t + sigma * np.sqrt(delta_t) * z)


                # 如果i属于敲出观察日，则需要判断是否敲出，计算模拟数据中有多少笔需要敲出

                if i in KnockOutTradeStep:

                    knock_out_keeper += (s_next >= knock_out_price)

                    current_knock_out = np.sum(knock_out_keeper != 0) - acc_knock_out_times

                    acc_knock_out_times += current_knock_out

                    # print(f"当前是MC的第{i}步需要进行敲出观察,本次敲出{current_knock_out}笔，累计敲出{acc_knock_out_times}笔")

                    cost_total = (deposit * np.exp(r * i*delta_t) - deposit) * current_knock_out  # i是估值日到敲出观察日对应的步数，由于是一天一步，i/T_trading也表示合约提前敲出的持续年限

                    # print('所有路径的在当前步上的保证金成本',cost_total)

                    # 计算本步敲出收益

                    amount_total = nominal * KnockOutActualStep[KnockOutTradeStep.index(i)] / 365 * (

                            knockout_Coupon + pay_Coupon) * current_knock_out - cost_total

                    # print('计息公式：',amount_total+cost_total,nominal,KnockOutActualStep[KnockOutTradeStep.index(i)],knockout_Coupon+pay_Coupon,current_knock_out)

                    discount = np.exp(-r * i*delta_t)

                    acc_income += amount_total * discount

                # 敲入统计

                knock_in_keeper += (s_next < knock_in_price)

                # print(knock_in_keeper)



            # 处理未敲出（knock_out_keeper为0的位置）且未敲入（knock_in_keeper为0的位置）

            lst1 = s_next[(knock_out_keeper == 0) & (knock_in_keeper == 0)]

            if len(lst1) != 0:  # 计算有几笔属于未敲出且未敲入

                cost_total1 = (deposit * np.exp(r * ResidualYears) - deposit) * len(lst1)

                abs_amount = nominal * ResidualYears * (knockout_Coupon + pay_Coupon) * len(lst1) - cost_total1

                discount = np.exp(-r * ResidualYears)

                acc_income += abs_amount * discount



            # 处理未敲出（knock_out_keeper为0的位置）且已敲入（knock_in_keeper不为0的位置）

            lst2 = s_next[(knock_out_keeper == 0) & (knock_in_keeper != 0)]

            if len(lst2) != 0:  # 计算有几笔属于未敲出且已敲入

                cost2 = (deposit * np.exp(

                    r * ResidualYears) - deposit)  # 这里不要乘以len(lst2)，因为下面方法会np.sum()，相当于乘以len(lst2)

                abs_amount = np.sum(nominal * (-np.maximum(0, 1-lst2/s0)) + pay_Coupon * ResidualYears) - cost2

                acc_income += abs_amount * discount

            pv = acc_income / numpath

        # 投资者的PV(雪球期权的买方PV)，单位元。

        return pv
    
    
    def calc_Delta(self,type_name='AutoCall_virtual'):
        ds = 0.01
        price_now = self.s0_now
        if type_name == 'AutoCall_virtual':
            self.s0_now = price_now*(1+ds)
            pv1 = self.AutoCall_virtual_pv()
            self.s0_now = price_now*(1-ds)
            pv2 = self.AutoCall_virtual_pv()
            delta = (pv1-pv2)/(2*ds*float(price_now))
        elif type_name == 'AutoCall_true':
            self.s0_now = price_now*(1+ds)
            pv1 = self.AutoCall_true_pv()
            self.s0_now = price_now*(1-ds)
            pv2 = self.AutoCall_true_pv()
            delta = (pv1-pv2)/(2*ds*float(price_now))
                
        return delta


    def calc_gamma(self,a):
        ds = 0.01

        price_now = a
        print(price_now)
        self.s0_now=price_now*(1+ds)
        delta1 = self.calc_Delta('AutoCall_virtual')
        self.s0_now = price_now*(1-ds)
        delta2 = self.calc_Delta('AutoCall_virtual')
        gamma=(delta1-delta2)/(2*ds*float(price_now))
        return gamma





if __name__ == "__main__":
        
    parm_dict={'s0':1,                     #期初价格
               's0_now':1,                 #标的资产价格
               'nominal':1,                #名义本金
               'deposit':0,                #保证金
               'knockin':0.85,             #敲入比例
               'knockout':1.03,            #敲出比例
               'begin_date':'2022-01-04',  #开始日期
               'val_date':'2022-01-04',    #估值日期
               'pay_Coupon':0,         #返息率（后端收益率(年化)），卖方根据合约存续天数，以名义本金和实际存续天数为基础，按返息率计息支付给买方
               'knockout_Coupon':0.2, #敲出收益率
               'knockout_month':4,    #封闭日
               'ratio':0.0,              #保底比例
               'r':0.03,               #无风险收益率
               'q':0.0,                #期货贴水率（年化）
               'sigma':0.13,           #标的波动率（年化）
               'numpath':300000,       #MC模拟次数
               'status':0,             #雪球当前状态，0表示当前未敲入，非0表示已经敲入
               'T_trading':252,         #表示一年的交易日天数
               'term':1,
               'path':'trade_days.xlsx',
               'StepLength':21
               }


    KnockOutDate = pd.read_excel("C:/Users/win10/Desktop/KnockOutDays.xlsx")
    # KnockOutDate['第i个月止盈观察日'] = KnockOutDate['第i个月止盈观察日'].dt.date
    KnockOutDate = list(KnockOutDate['第i个月止盈观察日'])
    parm_dict['KnockOutDate'] = KnockOutDate
    test = SnowBall(parm_dict)
    print(test.AutoCall_virtual_pv())
    print(test.calc_Delta())



    def draw_Delta(parm_dict, typename):
        import numpy as np
        delta_df = pd.DataFrame(columns=["s0_now", "Delta", 'Gamma'])
        lst1 = []
        lst2 = []
        lst3 = []
        parm_dic1 = parm_dict.copy()

        if typename == "virtual":
            for i in np.arange(0.6, 1.2, 0.01):
                s0_now = np.round(i, 2)
                lst1.append(s0_now)
                parm_dic1["s0_now"] = s0_now
                result = SnowBall(parm_dic1)
                delta = result.calc_Delta('AutoCall_virtual')
                gamma = result.calc_gamma(i)
                lst2.append(delta)
                lst3.append(gamma)
        delta_df['s0_now'] = lst1
        delta_df["Delta"] = lst2
        delta_df["Gamma"] = lst3

        return delta_df

    result=draw_Delta(parm_dict,'virtual')
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    ax1.plot(result['s0_now'],result['Gamma'],label="Gamma")
    ax1.set_title('Gamma')

    ax2.plot(result['s0_now'],result["Delta"],label="Delta")
    ax2.set_title('Delta')
    plt.show()
    plt.legend()



    
    
