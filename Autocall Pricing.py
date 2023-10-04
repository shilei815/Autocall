# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:31:26 2023

@author: Lei
"""
import numpy as np
import datetime as dt
import pandas as pd
from datetime import datetime
from dateutil import parser  
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Function to get list of valid trade dates from excel file 
def get_trade_date_lst():

    # Get the latest trading calendar
    # Load trade date excel file
    try:

        df_tradedays = pd.read_excel('trade_days.xlsx')

    except:
    # If file not in current dir, load from parent dir  
        path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))

        df_tradedays = pd.read_excel(path+'\\trade_days.xlsx')

    trade_date_lst= list(df_tradedays.iloc[:,0])

    # Return list of trade dates
    return trade_date_lst

# Function to get number of trading days between two dates
def get_delta_trade_days(begin, end):

    ''' Calculate the number of trading days between begin and end

    begin/end: format is yyyy-mm-dd.

    '''
    # Call function to get valid trade date list
    trade_date_lst = get_trade_date_lst()

    # Check if begin and end dates are in trade date list
    # If not, adjust dates to valid trade dates
    if begin in trade_date_lst and end in trade_date_lst:
        return trade_date_lst.index(end) - trade_date_lst.index(begin)
    else:
    # Logic to adjust dates to valid trade dates
        if not begin in trade_date_lst:
            for i in range(20):
                begin_new = dt.datetime.strftime(dt.datetime.strptime(begin, "%Y-%m-%d") - dt.timedelta(days=i),
                                                 "%Y-%m-%d")  # if begin is holiday, subtract 1 day until it is a trading day.
                if begin_new in trade_date_lst:
                    begin = begin_new
                    break
        if not end in trade_date_lst:
            for i in range(20):
                end_new = dt.datetime.strftime(dt.datetime.strptime(end, "%Y-%m-%d") - dt.timedelta(days=i),
                                               "%Y-%m-%d") # if end is holiday, subtract 1 day until it is a trading day.
                if end_new in trade_date_lst:
                    end = end_new
                    break
    # Return number of trading days
        return trade_date_lst.index(end) - trade_date_lst.index(begin)

def GetActualDaysBetweenTwoDate(begin,end):
    ''' Get actual number of days between begin and end
    begin/end: format is yyyy-mm-dd.
    '''
    begin1 = dt.datetime.strptime(begin,'%Y-%m-%d')
    end1 = dt.datetime.strptime(end,'%Y-%m-%d')
    return (end1-begin1).days


# Class to model autocall option and price using Monte Carlo simulation

class SnowBall:
    
    # Constructor and initialization
    def __init__(self,parm_dict):
    # Initialize attributes from parameter dictionary
        self.s0 = parm_dict.get('s0',0)                             # initial asset price
        self.s0_now = parm_dict.get('s0_now',0)                     # current asset price
        self.deposit=parm_dict.get('deposit',0)
        self.nominal = parm_dict.get('nominal',0)                   # notional principal
        self.knockin = parm_dict.get('knockin',0)                   # knockin level
        self.knockout = parm_dict.get('knockout')                   # knockout level
        self.knockout_Coupon = parm_dict.get('knockout_Coupon',0)   # knockout return
        self.knockout_month=parm_dict.get('knockout_month',0)       # lockout month
        self.r = parm_dict.get('r',0)                               # risk-free rate
        self.sigma = parm_dict.get('sigma',0)                       # volatility of underlying (annualized) 
        self.pay_Coupon=parm_dict.get('pay_Coupon',0)              
        self.numpath = parm_dict.get('numpath',0)                   # number of MC simulations
        self.q = parm_dict.get('q',0)                               # futures repo rate (annualized)
        self.T_trading = parm_dict.get('T_trading',240)             # number of trading days per year        
        self.term = parm_dict.get('term',1)                         # option term (years)
        self.StepLength = parm_dict.get('StepLength',21)            # number of days between observation dates
        self.val_date = parm_dict.get('val_date')                   # valuation date
        self.begin_date = parm_dict.get('begin_date')               # start date of the option
        self.KnockOutDate = parm_dict.get('KnockOutDate',"-")       # knockout observation dates
        self.path = parm_dict.get('path')

    # Function to return theoretical present value  
    def AutoCall_virtual_pv(self):
        '''Theoretically calculate the pv of the snowball option'''
        
        # Assign class attributes to local variables for faster processing
        r = self.r
        q = self.q
        sigma = self.sigma
        knockout_Coupon = self.knockout_Coupon
        numpath = self.numpath
        ResidualYears = self.term
        nominal = self.nominal
        acc_income = 0  
        knock_in_price = round(self.knockin * self.s0, 2)  
        knock_out_price = round(self.knockout * self.s0, 2)  
        s_next = np.ones(numpath) * self.s0_now  
        T_trading = self.T_trading
        Residualdays = int(self.term * self.T_trading)
        StepLength=self.StepLength
        KnockOutTradeStep = list(range(StepLength, Residualdays+1, StepLength))
        knock_in_keeper = np.zeros(numpath)  
        knock_out_keeper = np.zeros(numpath)   
        acc_knock_out_times = 0  
        begin_date=self.begin_date
        term=self.term
        knockout_month=self.knockout_month
        lock_date = (datetime.strptime(begin_date, '%Y-%m-%d') + relativedelta(months=knockout_month)).strftime('%Y-%m-%d')
        t_trade_days = get_delta_trade_days(begin_date, lock_date)

    # Monte Carlo simulation loop
        for i in range(1, Residualdays + 1):
            
            # Generate random walks for asset price
            # Monte Carlo simulation
            
            z = np.random.standard_normal(numpath)
            s_next = s_next * np.exp((r - q - 0.5 * np.square(sigma)) * ResidualYears / Residualdays + sigma * np.sqrt(
                ResidualYears / Residualdays) * z)

            if i in range(1,t_trade_days+1):
                lock_state=0
            else:
                lock_state=1

            # Check for knockout
            
            if i in KnockOutTradeStep and lock_state==1:
                knock_out_keeper += (s_next >= knock_out_price)
                current_knock_out = np.count_nonzero(knock_out_keeper != 0) - acc_knock_out_times
                acc_knock_out_times += current_knock_out
                # Calculate knockout payout
                amount_total = nominal * (i / T_trading) * (knockout_Coupon) * current_knock_out

                # print('计息公式:',amount_total+cost_total,nominal,KnockOutActualStep[KnockOutTradeStep.index(i)],knockout_Coupon+pay_Coupon,current_knock_out)
                discount = np.exp(-r * i / T_trading)
                acc_income += amount_total * discount
                
            # Check for knockin
            knock_in_keeper += (s_next < knock_in_price)

        # Handle paths not knocked out and not knocked in
        lst1 = s_next[(knock_out_keeper == 0) & (knock_in_keeper == 0)]
        if len(lst1) != 0:  
            abs_amount = nominal * term * (knockout_Coupon) * len(lst1)
            discount = np.exp(-r * ResidualYears)
            acc_income += abs_amount * discount

        # Handle paths not knocked out but knocked in
        lst2 = s_next[(knock_out_keeper == 0) & (knock_in_keeper != 0)]
        if len(lst2) != 0:
            abs_amount = np.sum(nominal * (-np.maximum(0, 1 - lst2 / self.s0)))
            acc_income += abs_amount * discount
            
        # Return present value
        return acc_income / numpath
        

    
    # Function to return true present value
    def AutoCall_true_pv(self):
        '''Calculate the true pv of the option'''
        
        # Simulation logic similar to above
        # Get parameters
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


        '''Check if s0_now has knocked out on val_date if val_date is after lockout period''' 

        knock_out_status = 0  
        if type(KnockOutDate) == str:
            KnockOutDate_lst = KnockOutDate.split(',')
            KnockOutDate_lst.sort(reverse=False)  
        else:
            KnockOutDate_lst = KnockOutDate


        if GetActualDaysBetweenTwoDate(lock_date,val_date) > 0:
            if val_date in KnockOutDate_lst:
                if s0_now >= knock_out_price:
                    knock_out_status = 1


        if knock_out_status == 1:

            pv = s0_now
        else:

            # Calculate dates
            
            # Get KnockOutTradeStep - number of trading days between each knockout observation date and val_date

            KnockOutTradeStep = []

            for date in KnockOutDate_lst:
                trade_days_between = get_delta_trade_days(val_date, date)
                if trade_days_between >= 0:  
                    KnockOutTradeStep.append(trade_days_between)

            # Get KnockOutActualStep - number of calendar days between each knockout observation date and begin_date

            KnockOutActualStep = []

            for date in KnockOutDate_lst:

                actual_days_between = GetActualDaysBetweenTwoDate(val_date, date)

                if actual_days_between >= 0:

                    KnockOutActualStep.append(actual_days_between)


            # Length adjustment

            KnockOutActualStep = KnockOutActualStep[-len(KnockOutTradeStep):]



            # Monte Carlo simulation
            
            Residualdays = get_delta_trade_days(val_date, end_date)  
            Days = GetActualDaysBetweenTwoDate(val_date, end_date)
            ResidualYears = Days / 365
            n = 1 * Residualdays  
            delta_t = ResidualYears/n

            knock_out_keeper = np.zeros(numpath)
            acc_knock_out_times = 0

            knock_in_keeper = np.zeros(numpath)

            s_next = np.ones(numpath) * s0_now

            acc_income = 0

            for i in range(1, n + 1):

                z = np.random.standard_normal(numpath)
                s_next = s_next * np.exp((r - q - 0.5 * np.square(sigma)) * delta_t + sigma * np.sqrt(delta_t) * z)

                if i in KnockOutTradeStep:
                
                    # Check for knockout
                    knock_out_keeper += (s_next >= knock_out_price)
                    current_knock_out = np.sum(knock_out_keeper != 0) - acc_knock_out_times
                    acc_knock_out_times += current_knock_out
                    
                    # Calculate knockout payout
                    cost_total = (deposit * np.exp(r * i*delta_t) - deposit) * current_knock_out  
                    amount_total = nominal * KnockOutActualStep[KnockOutTradeStep.index(i)] / 365 * (knockout_Coupon + pay_Coupon) * current_knock_out - cost_total
                    discount = np.exp(-r * i*delta_t)
                    acc_income += amount_total * discount

                # Check for knockin
                knock_in_keeper += (s_next < knock_in_price)
                

            # Handle paths not knocked out and not knocked in
            lst1 = s_next[(knock_out_keeper == 0) & (knock_in_keeper == 0)]
            if len(lst1) != 0:
                cost_total1 = (deposit * np.exp(r * ResidualYears) - deposit) * len(lst1)
                abs_amount = nominal * ResidualYears * (knockout_Coupon + pay_Coupon) * len(lst1) - cost_total1
                discount = np.exp(-r * ResidualYears)
                acc_income += abs_amount * discount
                
            # Handle paths not knocked out but knocked in
            lst2 = s_next[(knock_out_keeper == 0) & (knock_in_keeper != 0)]
            if len(lst2) != 0:
                cost2 = (deposit * np.exp(r * ResidualYears) - deposit) 
                abs_amount = np.sum(nominal * (-np.maximum(0, 1-lst2/s0)) + pay_Coupon * ResidualYears) - cost2
                acc_income += abs_amount * discount

            pv = acc_income / numpath

        return pv
    
    # Function to calculate delta
    def calc_Delta(self,type_name='AutoCall_virtual'):
    
    # Bump up/down asset price 
    # Call pricing functions to get bumped pvs
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

    # Function to calculate gamma
    def calc_gamma(self,a):
        ds = 0.01

    # Call calc_Delta twice with bumped price
    # Return gamma
        price_now = a
        
        self.s0_now=price_now*(1+ds)
        delta1 = self.calc_Delta('AutoCall_virtual')
        
        self.s0_now = price_now*(1-ds)
        delta2 = self.calc_Delta('AutoCall_virtual')
        
        gamma=(delta1-delta2)/(2*ds*float(price_now))
        return gamma



if __name__ == "__main__":
     
    # Create parameter dictionary
    parm_dict={'s0':1,                     # initial price
               's0_now':1,                 # current price
               'nominal':1,                # notional principal 
               'deposit':0,                # margin
               'knockin':0.85,             # knockin level  
               'knockout':1.03,            # knockout level
               'begin_date':'2022-01-04',  # start date
               'val_date':'2022-01-04',    # valuation date
               'pay_Coupon':0,             # fixed return (annualized) 
               'knockout_Coupon':0.2,      # knockout return
               'knockout_month':4,         # lockout month
               'ratio':0.0,                # floor ratio
               'r':0.03,                   # risk-free rate
               'q':0.0,                    # futures repo rate  
               'sigma':0.13,               # volatility
               'numpath':300000,           # number of MC sims
               'status':0,                 # 0 = not knocked in, 1 = knocked in  
               'T_trading':252,            # number of trading days per year
               'term':1,                   # option term (years)
               'path':'trade_days.xlsx',   
               'StepLength':21            
               }


    KnockOutDate = pd.read_excel("C:/Users/win10/Desktop/KnockOutDays.xlsx")
    KnockOutDate = list(KnockOutDate['ith stop profit observation date'])
    parm_dict['KnockOutDate'] = KnockOutDate
    
    # Create parameter dictionary
    test = SnowBall(parm_dict)
    
    # Call pricing functions
    print(test.AutoCall_virtual_pv())
    print(test.calc_Delta())



    def draw_Delta(parm_dict, typename):
        
        # Draw delta and gamma graphs
        
        delta_df = pd.DataFrame(columns=["s0_now", "Delta", 'Gamma'])
        lst1 = []
        lst2 = []
        lst3 = []
        parm_dic1 = par
        
        
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