import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import streamlit as st
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statistics import mean 

class StatergyAnalysis:
    
    def __init__(self, csv_data, is_dataframe=0, number=150000, customerPLBook = False):
        self.csv_data = self.new_csv(csv_data , is_dataframe)
        self.initial_investment = number
        self.daily_returnts = None
        self.monthly_returns = None
        self.weekly_returns = None
        self.weekday_returns = None
        self.yearly_returns = None
        _,_,_,_,_ =  self.analysis()
        #self.daily_ana()
        self.daily_equity = self.csv_data.groupby('Day')['equity_curve'].last()
        #self.daily_equity_curve = self.daily_equity_curve[['equity_curve']]
        self.equity_curve_value = self.csv_data['pnl_cumulative_absolute'] + self.initial_investment
        self.risk_free_rate = 0.07
        self.equity_PctChange = None
        self.annual_std = 0
        self.annual_mean = 0
        self.daily_annual_mean = 0
        self.daily_annual_std = 0
        self.drawdown_max, self.drawdown_pct = self.drawdown(customerPLBook=customerPLBook)
        self.daily_equity_Curve(customerPLBook=customerPLBook)
        self.num_wins = self.num_profit(self.csv_data)
        self.numTrades = len(self.csv_data)
        self.minProfits = []
        self.MaxProfits =[] 
        self.hit = round((self.num_wins/self.numTrades*100), 2)
        self.Hit_Daywise = self.HIT_day()
        self.long = self.num_tradeType("long")
        self.short = self.num_tradeType("short")
        self.profit_factor, self.neg, self.pos = self.ProfitFactor()
        self.profit = self.max_profit(self.csv_data)
        self.loss = self.max_profit(self.csv_data, i=4)
        
    def new_csv(Self, filepath, is_dataframe):
        if is_dataframe == 0:
            data = pd.read_csv(filepath)
        else:
            data = pd.DataFrame(filepath)
            
        if 'EN_TIME' in data.columns:
            data.rename(columns={'EN_TIME': 'entry_timestamp'}, inplace=True)
        if 'P&L' in data.columns:
            data.rename(columns={'P&L': 'pnl_absolute'}, inplace=True)
        if 'Equity Curve' in data.columns:
            data.rename(columns={'Equity Curve': 'equity_curve'}, inplace=True)
        if 'EN_TT' in data.columns:
            data.rename(columns={'EN_TT': 'entry_transaction_type'}, inplace=True)
        if 'cumulative_pnl_absolute' in data.columns:
            data.rename(columns={'cumulative_pnl_absolute': 'pnl_cumulative_absolute'}, inplace=True)
        if 'Drawdown_%' in data.columns:
            data.rename(columns={'Drawdown_%': 'drawdown_percentage'}, inplace=True)
        if 'Drawdown %' in data.columns:
            data.rename(columns={'Drawdown %': 'drawdown_percentage'}, inplace=True)
        data['entry_transaction_type'] = data['entry_transaction_type'].replace({'BUY': 0, 'SELL': 1})

        data = data.dropna(subset=['pnl_absolute'])
        if 'Day' not in data.columns:
            data['date'] = pd.to_datetime(data['entry_timestamp'])
            start = data['date'].iloc[0]
            end = data['date'].iloc[-1]
            if start > end:
                data = data.iloc[::-1].reset_index(drop=True)    
            data = data.reset_index()
            data = data.drop(columns=['entry_timestamp'])

            data['Day'] = pd.to_datetime(data.date,format = '%Y-%m')
            data['Week'] = pd.to_datetime(data.date,format = '%dd-%m')
            data['Month'] = pd.to_datetime(data.date,format = '%Y-%m')
            data['Year'] = pd.to_datetime(data.date,format = '%Y-%m')
            data['weekday'] = pd.to_datetime(data.date,format = '%a')
            
            data['Day'] = data['Day'].dt.strftime('%Y-%m-%d')
            data['Week'] = data['Week'].dt.strftime('%Y-%U')
            data['Month'] = data['Month'].dt.strftime('%Y-%m')
            data['Year'] = data['Year'].dt.strftime('%Y')
            data['weekday'] = data['weekday'].dt.strftime('%a')
        return data
        
    def daily_equity_Curve(self, customerPLBook=False):
        if customerPLBook:
            daily_equity_curve = self.daily_returnts['cum_pnl'] + self.initial_investment
        else:
            daily_equity_curve = self.csv_data.groupby('Day')['equity_curve'].last()
        self.equity_PctChange = daily_equity_curve.pct_change().dropna()
        self.daily_annual_mean = self.equity_PctChange.mean() * np.sqrt(252)
        self.daily_annual_std = self.equity_PctChange.std() * np.sqrt(252) 

    def yearlyVola(self, customerPLBook=False):
        if customerPLBook:
            equity = self.csv_data['equity_calculated']
        else:
            equity = self.csv_data['pnl_cumulative_absolute'] + self.initial_investment
        equity_PctChange = equity.pct_change().dropna()
        self.annual_std = equity_PctChange.std() * np.sqrt(252) 
        self.annual_mean = equity_PctChange.mean() * np.sqrt(252) 
        return round(self.annual_std * 100, 2)

    def analysis(self):
        daily_returns = self.csv_data.groupby('Day').sum(numeric_only = True)
        daily_returns['cum_pnl'] = daily_returns['pnl_absolute'].cumsum()
        daily_returns['roi'] = round((daily_returns['cum_pnl']/self.initial_investment)*100,2)
        daily_analysis = daily_returns[['pnl_absolute', 'cum_pnl', 'roi']]
        self.daily_returnts = daily_analysis

        Monthly_returns = self.csv_data.groupby('Month').sum(numeric_only = True)
        Monthly_returns['cum_pnl'] = Monthly_returns['pnl_absolute'].cumsum()
        Monthly_returns['roi'] = round((Monthly_returns['cum_pnl']/self.initial_investment)*100,2)
        monthly_analysis = Monthly_returns[['pnl_absolute', 'cum_pnl', 'roi']]
        self.monthly_returns = monthly_analysis
        
        weekday_returns = self.csv_data.groupby('weekday').sum(numeric_only = True)
        #weekday_returns['pnl_absolute'] = weekday_returns['pnl_absolute'].abs()
        weekday_returns = weekday_returns[weekday_returns['pnl_absolute'] != 0]
        weekday_returns[['pnl_absolute']]
        self.weekday_returns = weekday_returns[['pnl_absolute']]

        weekly_returns = self.csv_data.groupby('Week').sum(numeric_only = True)
        weekly_returns['cum_pnl'] = weekly_returns['pnl_absolute'].cumsum()
        weekly_returns[['pnl_absolute','cum_pnl']]
        self.weekly_returns = weekly_returns[['pnl_absolute','cum_pnl']]

        yearly_returns = self.csv_data.groupby('Year').sum(numeric_only = True)
        yearly_returns['cum_pnl'] = yearly_returns['pnl_absolute'].cumsum()
        yearly_returns[['pnl_absolute', 'cum_pnl']]
        self.yearly_returns =yearly_returns[['pnl_absolute', 'cum_pnl']]

        return daily_analysis, monthly_analysis, weekday_returns, weekly_returns, yearly_returns   

    def max_profit(self, returns, i=1):
        if i ==1:
            max_profits = returns['pnl_absolute'].max()
            max_profitable_day = returns['pnl_absolute'].idxmax()
        else:
            max_profits = returns['pnl_absolute'].min()
            max_profitable_day =  returns['pnl_absolute'].idxmin()
        maxi = [max_profits, max_profitable_day]
        #self.MaxProfits = maxi
        return maxi

    def min_profit(self, returns):
        min_profitable_day = returns['pnl_absolute'].min()
        min_profit_day =  returns['pnl_absolute'].idxmin()
        #self.minProfits = [min_profitable_day, min_profit_day] 
        return [min_profitable_day, min_profit_day] 
    
    def Sharpe(self):
        sharpe_ratio = (self.daily_annual_mean*np.sqrt(252) - self.risk_free_rate) / self.daily_annual_std
        return round(sharpe_ratio, 2)
    
    def Calmar(self):
        calmar_ratio = self.daily_annual_mean*np.sqrt(252) / self.drawdown_pct * -100 
        return round(calmar_ratio, 2)
    
    def Sortino(self):
        downside_returns = np.where(self.equity_PctChange < 0, self.equity_PctChange, 0)
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (self.daily_annual_mean*np.sqrt(252) - self.risk_free_rate) / downside_deviation
        return round(sortino_ratio, 2)
    
    def max_consecutive(self, daily_returns, quant):
  
        if quant ==1:
            positive_mask = daily_returns['pnl_absolute'] > 0
        else:
            positive_mask = daily_returns['pnl_absolute']  < 0
        
        grouped = (positive_mask != positive_mask.shift()).cumsum()
        positive_counts = positive_mask.groupby(grouped).cumsum()
        return positive_counts.max()
    
    def win_rate(self, daily_returns):
        wins = daily_returns[daily_returns['pnl_absolute']>0]   
        return round(len(wins)/len(daily_returns)*100, 2)
    
#Another Useless function
    def winCount(self, daily_returns, i):
        wins = daily_returns[daily_returns['pnl_absolute']>0]
        loss = daily_returns[daily_returns['pnl_absolute']<0]
        
        if i >0:
            return len(wins)
        else:
            
            return len(loss)
    
    def date_calc(self, day=0, returns=None):
        if returns is None:
            returns = self.daily_returnts
            
        last_date = datetime.strptime(returns.index[-1], '%Y-%m-%d')
        start_date = last_date - relativedelta(days=day)
        for i in returns.index:
            date =  datetime.strptime(i, '%Y-%m-%d')
            if date >= start_date:
                start_date = date
                break
        new_date_str = start_date.strftime('%Y-%m-%d')
        index_number = returns.index.get_loc(new_date_str)
        ind = returns['cum_pnl'].iloc[-(len(returns)-index_number)]
        return new_date_str
        
    def Treturns(self, day=0, returns=None):
        if returns is None:
            returns = self.daily_returnts
            
        new_date_str = self.date_calc(day=day, returns=returns)
        final_equity_value = returns['cum_pnl'].iloc[-1]
        start_equity_value = returns.loc[new_date_str, 'cum_pnl']
        gain = final_equity_value - start_equity_value
        return gain, round(gain*100/self.initial_investment, 2)


    def avgReturns(self, daily_returns):
        daily_returns['returns'] = daily_returns['cum_pnl']/self.initial_investment *100
        avg_returns = daily_returns['cum_pnl'].mean()
        avg_returns_pct = daily_returns['returns'].mean()
        return [round(avg_returns, 2), round(avg_returns_pct, 2)]
    
    def drawdown(self, customerPLBook=False ):
    
        if customerPLBook:
            self.csv_data['pnl_cumulative'] = self.csv_data['pnl_absolute'].cumsum()
            self.csv_data['equity_calculated'] = self.csv_data['pnl_cumulative'] + self.initial_investment
            self.csv_data['equity_cum_max'] = self.csv_data['equity_calculated'].cummax()
            self.csv_data['drawdown'] = self.csv_data['equity_calculated'] - self.csv_data['equity_cum_max']        
            self.csv_data['drawdown_pct'] = (self.csv_data['drawdown']/self.csv_data['equity_cum_max'])*100
        else:
            self.csv_data['equity_cum_max'] = self.csv_data['equity_curve'].cummax()
            self.csv_data['drawdown'] = self.csv_data['equity_curve'] - self.csv_data['equity_cum_max']
            self.csv_data['drawdown_pct'] = self.csv_data['drawdown_percentage']
            
        return round(self.csv_data['drawdown'].min(), 2), round(self.csv_data['drawdown_pct'].min(), 2)
    
    def daily_returns_hist(self, daily_returns):
        fig1, ax1 = plt.subplots(figsize=(10, 2))  
        ax1.bar(daily_returns.index, daily_returns['pnl_absolute'])
        ax1.set_xticks([])
        ax1.set_yticks([])

        fig2, ax2 = plt.subplots(figsize=(10, 5))  
        ax2.bar(daily_returns.index, daily_returns['cum_pnl'])
        ax2.set_xticklabels([])
        
        return fig1, fig2

    def roi(self, monthly_returns=None):
        if monthly_returns is None:
            monthly_returns = self.monthly_returns
        ROI = monthly_returns[['cum_pnl']].iloc[-1]
        ROI_perct = round((ROI.values[0]/self.initial_investment)*100,2)
        return round(ROI.values[0], 2), round(ROI_perct, 2)

    def num_profit(self, returns):
        return sum(returns['pnl_absolute'] > 0)

    def num_loss(self, returns, i):
        if i == -1:
            return sum(returns['pnl_absolute'] < 0)
        else:
            return sum(returns['pnl_absolute'] >0)
        
# try to remove this function
    def trading_num(self, returns):
        return len(returns)

    def compare_hist(self, returns, num, Period):
        
        df = pd.DataFrame()
        df["Value"] = num
        profit = []
        loss = []
        for value in num:
            profit.append(sum(returns['pnl_absolute'] > value))
            loss.append((sum(returns['pnl_absolute'] < -1 * value))) 
       
        df["Profit"] = profit
        df["Loss"] = loss

        n= len(num)
        r = np.arange(n) 
        width = 0.25

        fig, ax = plt.subplots()
        ax.bar(r, profit, color = 'b', width = width, label='Profit')
        ax.bar(r + width, loss, color = 'r', width = width, label='Loss') 
        ax.set_xlabel("Value") 
        ax.set_ylabel(f"Number of {Period}") 
        ax.set_xticks(r + width / 2)
        ax.set_xticklabels(num)
        ax.legend() 
  
        df.set_index('Value', inplace=True)
        return fig, df
        
    def round_calc(self, retrurn):
        profit = np.zeros(12)
        for r in retrurn:
            val = int(r // 1000 + 6)
            if val < 0:
                val = 0
            if val> 11:
                val = 11
            profit[val] +=1
        return profit
            
    
    def freq_hist(self, returns, num):
        profit = self.round_calc(returns['pnl_absolute'])
        num.insert(0, "")  
        num.append("")  
        
        n = len(profit)
        r = np.arange(1, n + 1)  
        width = 0.75

        fig, ax = plt.subplots()
        ax.bar(r - 0.5, profit, color='b', width=width, align='center', label='Profit')
        for index, value in enumerate(profit):
            if value > 0:
                ax.text(index + 0.5, value, str(value), ha='center')
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_xticks(np.arange(len(num)))
        ax.set_xticklabels(num, rotation=45)
        ax.legend()

        return fig

    def HIT(self):
        return round((self.num_wins/self.numTrades*100), 2)
    
    def HIT_day(self):
        num_wins = self.num_loss(self.daily_returnts, 1)
        return round(num_wins/len(self.daily_returnts)* 100, 2) 
    
    # def relative_date(self, year, month, days):
    #     given_date = 
    def num_tradeType(self, quant):
        i = -1
        if quant == "short":
            i = 1
        elif quant == "long":
            i = 0
        else :
            return None
        
        trad = self.csv_data[self.csv_data['entry_transaction_type'] == i]
        return len(trad)
            
    def avgTrades(self, daily_returns):
        return round(len(self.csv_data)/len(daily_returns) , 2)
    
    def ProfitFactor(self):
        daily_positive = self.daily_returnts[self.daily_returnts['pnl_absolute'] > 0]['pnl_absolute'].sum()
        daily_neg = self.daily_returnts[self.daily_returnts['pnl_absolute'] < 0]['pnl_absolute'].sum()
        return round(daily_positive/daily_neg * -1 , 2), daily_neg, daily_positive
 
    def avgProfit(self, daily_returns=None, i=1):
        if daily_returns is None:
            daily_returns = self.csv_data
        wins = None
        if i == 1:
            wins = daily_returns[daily_returns['pnl_absolute']>0]
        else:
            wins = daily_returns[daily_returns['pnl_absolute'] <0]
        prof = wins['pnl_absolute'].tolist()
        return round(mean(prof), 2), len(wins)
        
    def htmap(self, days):        
        data = np.array(self.daily_returnts['pnl_absolute'].tolist())
        data = data[-1 * days:]
        m = 5 * int(len(data)/ 5)
        data = data[:m]
        data = np.reshape(data, (5, -1))
        line_width = 0.8
        linecolor = "White"
        box_width = int(m*2/25) + 2
    
        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v = [0,.15,.4,.5,0.6,.9,1.]
        l = list(zip(v,c))
        cm=LinearSegmentedColormap.from_list('rg',l, N=256)

        hm, ax = plt.subplots(figsize=(box_width,2), dpi=400)
        sn.heatmap(data=data, linecolor=linecolor, linewidths=line_width, cmap=cm, center=0, ax=ax) 

        st.write(hm)
        return box_width
      
    def Monthly_Roi_plot(self):
        
        fig, ax1 = plt.subplots()

    # Bar plot for monthly returns
        ax1.bar(self.monthly_returns.index.values, self.monthly_returns['cum_pnl'].values, color='b', alpha=0.6, label='Monthly Returns')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Monthly Returns', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(self.monthly_returns.index[::3])
        ax1.set_xticklabels(self.monthly_returns.index[::3], rotation=90)
    # Line plot for ROI%
        ax2 = ax1.twinx()
        ax2.plot(self.monthly_returns.index.values, self.monthly_returns['roi'].values, color='r', marker='o', label='ROI%')
        ax2.set_ylabel('ROI%', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
        return fig

