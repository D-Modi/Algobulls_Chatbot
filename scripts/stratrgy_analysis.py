import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import streamlit as st
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap

class StatergyAnalysis:

    def __init__(self, csv_filepath):
        self.csv_data = self.new_csv(csv_filepath)
        self.daily_returnts = None
        self.monthly_returns = None
        self.daily_ana()
        self.daily_equity = self.csv_data.groupby('Day')['equity_curve'].last()
        #self.daily_equity_curve = self.daily_equity_curve[['equity_curve']]
        self.equity_curve_value = self.csv_data['equity_curve'].tolist()
        self.risk_free_rate = 0.07
        #self.initial_investment = self.equity_curve_value[-1]
        self.initial_investment = 150000
        self.equity_PctChange = None
        self.annual_std = 0
        self.annual_mean = 0
        self.drawdown_max, self.drawdown_pct = self.drawdown()
        self.daily_equity_Curve()
        self.num_wins = self.num_profit(self.csv_data)
        self.numTrades = len(self.csv_data)

    
    def new_csv(Self, filepath):
        data = pd.read_csv(filepath)

        if 'EN_TIME' in data.columns:
            data.rename(columns={'EN_TIME': 'entry_timestamp'}, inplace=True)
        if 'P&L' in data.columns:
            data.rename(columns={'P&L': 'pnl_absolute'}, inplace=True)
        if 'Equity Curve' in data.columns:
            data.rename(columns={'Equity Curve': 'equity_curve'}, inplace=True)
        if 'EN_TT' in data.columns:
            data.rename(columns={'EN_TT': 'entry_transaction_type'}, inplace=True)
        if 'Drawdown %' in data.columns:
            data.rename(columns={'Drawdown %': 'drawdown_percentage'}, inplace=True)

        data = data.dropna(subset=['pnl_absolute'])
        
        data['date'] = pd.to_datetime(data['entry_timestamp'])
        data = data.sort_values(by='date')
        data = data.drop(columns=['entry_timestamp'])

        data['Day'] = pd.to_datetime(data.date,format = '%Y-%m')
        data['Week'] = pd.to_datetime(data.date,format = '%dd-%m')
        data['Month'] = pd.to_datetime(data.date,format = '%Y-%m')
        data['Year'] = pd.to_datetime(data.date,format = '%Y-%m')
        data['weekday'] = pd.to_datetime(data.date,format = '%a')\
        
        data['Day'] = data['Day'].dt.strftime('%Y-%m-%d')
        data['Week'] = data['Week'].dt.strftime('%Y-%U')
        data['Month'] = data['Month'].dt.strftime('%Y-%m')
        data['Year'] = data['Year'].dt.strftime('%Y')
        data['weekday'] = data['weekday'].dt.strftime('%a')
        return data
        
    def daily_equity_Curve(self):

        self.equity = self.daily_returnts['cum_pnl'] + self.initial_investment
        self.equity_PctChange = self.equity.pct_change().dropna()
        self.annual_mean = self.equity_PctChange.mean() * 252
        daily_mean = self.equity_PctChange.mean() 
        daily_std =  self.equity_PctChange.std()
        self.annual_std = self.equity_PctChange.std() * np.sqrt(252) 


    def yearlyVola(self):
        equity = self.csv_data['pnl_cumulative_absolute'] + self.initial_investment
        equity_PctChange = equity.pct_change().dropna()
        annual_std = equity_PctChange.std() * np.sqrt(252) 
        return round(annual_std * 100, 2)

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
        

        weekday_returns = self.csv_data.groupby('weekday').sum(numeric_only = True)
        #weekday_returns['pnl_absolute'] = weekday_returns['pnl_absolute'].abs()
        weekday_returns = weekday_returns[weekday_returns['pnl_absolute'] != 0]
        weekday_returns[['pnl_absolute']]

        weekly_returns = self.csv_data.groupby('Week').sum(numeric_only = True)
        weekly_returns['cum_pnl'] = weekly_returns['pnl_absolute'].cumsum()
        weekly_returns[['pnl_absolute','cum_pnl']]

        yearly_returns = self.csv_data.groupby('Year').sum(numeric_only = True)
        yearly_returns['cum_pnl'] = yearly_returns['pnl_absolute'].cumsum()
        yearly_returns[['pnl_absolute', 'cum_pnl']]

        return daily_analysis, monthly_analysis, weekday_returns, weekly_returns, yearly_returns 
    
    def daily_ana(self):
        daily_returns = self.csv_data.groupby('Day').sum(numeric_only = True)
        daily_returns['cum_pnl'] = daily_returns['pnl_absolute'].cumsum()
        daily_analysis = daily_returns[['pnl_absolute', 'cum_pnl']]
        self.daily_returnts = daily_analysis      

    def max_profit(self, returns):
        max_profits = returns['pnl_absolute'].max()
        max_profitable_day = returns['pnl_absolute'].idxmax()
        maxi = [max_profits, max_profitable_day]
        return maxi

    def min_profit(self, returns):
        min_profitable_day = returns['pnl_absolute'].min()
        min_profit_day =  returns['pnl_absolute'].idxmin()
        return [min_profitable_day, min_profit_day] 
    
    def Sharpe(self):
        sharpe_ratio = (self.annual_mean - self.risk_free_rate) / self.annual_std
        return round(sharpe_ratio, 2)
    
    def Calmar(self):
        calmar_ratio = self.annual_mean / self.drawdown_pct * -100
        return round(calmar_ratio, 2)
    
    def Sortino(self):
        downside_returns = np.where(self.equity_PctChange < 0, self.equity_PctChange, 0)
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (self.annual_mean - self.risk_free_rate) / downside_deviation
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

    def winCount(self, daily_returns, i):
        wins = daily_returns[daily_returns['pnl_absolute']>=0]
        if i >0:
            return len(wins)
        else:
            return len(daily_returns) - len(wins)
        
    def Treturns(self, t):
        cum_pnl = self.daily_returnts['cum_pnl'].tolist()
        cum_pnl = cum_pnl[-1*t:]
        ret = cum_pnl[-1] - cum_pnl[0]
        return ret, round(ret*100/self.initial_investment, 2)

    def avgReturns(self, daily_returns):
        daily_returns['returns'] = daily_returns['cum_pnl']/self.initial_investment *100
        avg_returns = daily_returns['cum_pnl'].mean()
        avg_returns_pct = daily_returns['returns'].mean()
        return round(avg_returns, 2), round(avg_returns_pct, 2)
    
    def drawdown(self):
        self.csv_data['cum_max'] = self.csv_data['equity_curve'].cummax()
        self.csv_data['drawdown'] = self.csv_data['equity_curve'] - self.csv_data['cum_max']
        self.csv_data['drawdown_pct'] = (self.csv_data['drawdown']/self.csv_data['cum_max'])*100
    
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

    def roi(self, monthly_returns):
        ROI = monthly_returns[['cum_pnl']].iloc[-1]
        ROI_perct = round((ROI.values[0]/150000)*100,2)
        return round(ROI.values[0], 2), round(ROI_perct, 2)

    def num_profit(self, returns):
        return sum(returns['pnl_absolute'] > 0)

    def num_loss(self, returns):
        return sum(returns['pnl_absolute'] < 0)

    def trading_num(self, returns):
        return len(returns)

    def compare_hist(self, returns, num, Quant):
        
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
        ax.set_ylabel(f"Number of {Quant}") 
        ax.set_xticks(r + width / 2)
        ax.set_xticklabels(num)
        ax.legend() 
  
        df.set_index('Value', inplace=True)
        return fig, df
    
    def freq_hist(self, returns, num):

        profit = []
        profit.append(sum(returns['pnl_absolute'] < num[0]))
        i = 1
        while i < len(num):
            profit.append(sum((returns['pnl_absolute'] > num[i-1]) & (returns['pnl_absolute'] < num[i])))
            i += 1
        profit.append(sum(returns['pnl_absolute'] > num[-1]))
        num.insert(0, "")  
        num.append("")  
        
        n = len(profit)
        r = np.arange(1, n + 1)  
        width = 0.75

        fig, ax = plt.subplots()
        ax.bar(r - 0.5, profit, color='b', width=width, align='center', label='Profit')
        for index, value in enumerate(profit):
            ax.text(index + 0.5, value + 1, str(value), ha='center')
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_xticks(np.arange(len(num)))
        ax.set_xticklabels(num, rotation=45)
        ax.legend()

        return fig

    def HIT(self):
        return round((self.num_wins/self.numTrades*100), 2)
    
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
        return round(daily_positive/daily_neg * -1 , 2)
    
    def trades_pie(self):
        Employee = ['Short', 'Long']
        Salary = [self.num_tradeType('short'), self.num_tradeType('long')]
        f, ax = plt.subplot()
        # Pie Chart
        plt.pie(Salary, labels=Employee,
                autopct='%1.1f%%', pctdistance=0.85)
        
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        
        # Adding Circle in Pie chart
        fig.gca().add_artist(centre_circle)
        plt.title('Trades')
        plt.show()

    def htmap(self):
        
        data = np.array(self.daily_returnts['pnl_absolute'].tolist())
        m = 5 * int(len(data)/ 5)
        data = data[:m]
        data = np.reshape(data, (5, -1))
        line_width = 0.8
        linecolor = "White"

        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v = [0,.15,.4,.5,0.6,.9,1.]
        l = list(zip(v,c))
        cm=LinearSegmentedColormap.from_list('rg',l, N=256)

        hm, ax = plt.subplots(figsize=(72,2), dpi=400)
        # plotting the heatmap 
        sn.heatmap(data=data, linecolor=linecolor, linewidths=line_width, cmap=cm, center=0, ax=ax) 
        st.write(hm)
      
       