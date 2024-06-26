# consecutoive days, max_profit, htmap, monthly returns and moi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import streamlit as st
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap
from statistics import mean 
from stratrgy_analysis import StatergyAnalysis

class merge_csv:
    
    def __init__(self, csv1, csv2):
        self.data1 = csv1
        self.data2 = csv2
        self.initial_investment = self.data1.initial_investment
        # self.daily_returns_combined = pd.concat([self.data1.daily_returnts, self.data2.daily_returnts]) # Only for "pnl_absolute"; not for cum_pnl
        # self.daily_returns_combined['cum_pnl'] = self.daily_returns_combined['pnl_absolute'].cumsum()
        self.daily_returns_combined = self.merged_df(self.data1.daily_returnts, self.data2.daily_returnts)
        self.combined_mean = (self.data1.annual_mean * self.data1.numTrades + self.data2.annual_mean * self.data2.numTrades )/ (self.data1.numTrades + self.data2.numTrades)
        self.combined_variance = (((self.data1.numTrades - 1) * self.data1.annual_std**2 + (self.data2.numTrades - 1) * self.data2.annual_std **2) +  self.data1.numTrades* (self.data1.annual_mean - self.combined_mean)**2 +  self.data2.numTrades * (self.data2.annual_mean - self.combined_mean)**2) / (self.data1.numTrades + self.data2.numTrades - 1)
        self.risk_free_rate = self.data1.risk_free_rate
        self.Yearly_Vola = round(self.combined_variance *100, 2)
        self.sharpe_ratio = round((self.combined_mean - self.risk_free_rate)/ self.combined_variance, 2)
        self.min_DDPct = min(self.data1.drawdown_pct, self.data2.drawdown_pct)
        self.Calmar_ratio = self.combined_mean / self.min_DDPct * -100
        #self.equity = pd.concat([self.data1.equity, self.data2.equity])
        transition_equity_change = (self.data2.csv_data['equity_curve'].iloc[0] - self.data1.csv_data['equity_curve'].iloc[-1]) / self.data1.csv_data['equity_curve'].iloc[-1]
        # self.equity_PctChange = self.data1.equity_PctChange.copy()
        # self.equity_PctChange.loc[len(self.equity_PctChange)] = transition_equity_change
        # self.equity_PctChange = self.equity_PctChange.append(self.data2.equity_PctChange, ignore_index=True)
        self.equity_PctChange = self.data1.equity_PctChange.copy()
        self.equity_PctChange = pd.concat([self.equity_PctChange, pd.Series([transition_equity_change])], ignore_index=True)
        self.equity_PctChange = pd.concat([self.equity_PctChange, self.data2.equity_PctChange], ignore_index=True)
        self.sortino_ratio = self.Sortino_Ratio()
        _,_ = self.data2.drawdown(i=1, max_eq=self.data1.csv_data['cum_max'].iloc[-1])
        self.drawdown_column = pd.concat([self.data1.csv_data[['drawdown', 'drawdown_pct']], self.data2.csv_data[['drawdown', 'drawdown_pct']]])
        self.drawdown_max = round(self.drawdown_column['drawdown'].min(), 2)
        self.drawdown_pct = round(self.drawdown_column['drawdown_pct'].min(), 2)
        self.HIT = round((self.data1.num_wins + self.data2.num_wins)/(self.data1.numTrades + self.data2.numTrades)* 100, 2)
        self.long = self.data1.long + self.data2.long
        self.short = self.data1.short +self.data2.short
        self.avgNumTrades = round((self.data1.numTrades + self.data2.numTrades)/(len(self.data1.daily_returnts) + len(self.data2.daily_returnts)), 2)
        self.ProfitFactor = round((self.data1.pos + self.data2.pos)/(self.data1.neg + self.data2.neg)* -1 , 2) 
        self.maxProfit = max([self.data1.profit, self.data2.profit], key=lambda lst: lst[0])
        self.maxLoss = min([self.data1.loss, self.data2.loss], key=lambda lst: lst[0])  
        self.monthly_ret = self.merged_df(self.data1.monthly_returns, self.data2.monthly_returns)
        self.csv_data_combined = pd.concat([self.data1.csv_data, self.data2.csv_data])
    def merged_df(self, df1, df2):
        dt = df1.index[-1]
        cumsum = df1['cum_pnl'].iloc[-1]
        result_df = df1.add(df2, fill_value=0)
      
        if 'cum_pnl' in result_df.columns:   
            result_df.loc[result_df.index > dt, 'cum_pnl'] += cumsum
        return result_df
    
    def max_min_Profit(self, i, returns1,returns2):
        MaxProfits1 = self.data1.max_profit(returns1, i)
        MaxProfits2 = self.data1.max_profit(returns2, i)
        if i == 1:
            if MaxProfits1[1] == returns1.index[-1] or MaxProfits2[1] == returns2.index[0]:
                if returns1.index[-1] == returns2.index[0]:
                    Maxprofits3 = returns1['pnl_absolute'].iloc[-1] + returns2['pnl_absolute'].iloc[0]
                    if max(MaxProfits1, MaxProfits2, Maxprofits3) == Maxprofits3:
                        return [Maxprofits3, returns1.index[-1]]
                    if MaxProfits2[1] == returns2[0]:
                        MP = self.data1.max_profit(returns2[1:]. i)
                        return max([MP, MaxProfits1], key=lambda lst: lst[0])
                    elif MaxProfits1[1] == returns1.index[-1] :
                        MP = self.data1.max_profit(returns1[:-1], i)
                        return max([MP, MaxProfits2], key=lambda lst: lst[0])
    
            return max([MaxProfits1, MaxProfits2], key=lambda lst: lst[0])
        else:
                if MaxProfits1[0] < MaxProfits2[0]:
                    return MaxProfits1
                else:
                    return MaxProfits2


    def Sortino_Ratio(self):
        downside_returns = np.where(self.equity_PctChange < 0, self.equity_PctChange, 0)
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return round((self.combined_mean - self.risk_free_rate) / downside_deviation, 2)
    
    def Comb_WinRate(self, daily_returns1, dailyreturns2):
        w1 = self.data1.win_rate(daily_returns1)
        w2 = self.data1.win_rate(dailyreturns2)
        tw = (w1 * len(daily_returns1) + w2* len(dailyreturns2))/(len(daily_returns1) + len(dailyreturns2))
        return round(tw, 2)
    
    def max_consecutive(self, daily_returns1, daily_returns2, w, quant):
        if w == 1:
            daily_returns = self.merged_df(daily_returns1, daily_returns2)
        else:
            daily_returns =  pd.concat([daily_returns1, daily_returns2]) 
        if quant ==1:
            positive_mask = daily_returns['pnl_absolute'] > 0
        else:
            positive_mask = daily_returns['pnl_absolute']  < 0
        
        grouped = (positive_mask != positive_mask.shift()).cumsum()
        positive_counts = positive_mask.groupby(grouped).cumsum()
        return positive_counts.max()
    
    def Treturns(self, t):
        cum_pnl = self.daily_returns_combined['cum_pnl'].tolist()
        cum_pnl = cum_pnl[-1*t:]
        ret = cum_pnl[-1] - cum_pnl[0]
        return ret, round(ret*100/self.initial_investment, 2)
        
    def htmap(self, days):        
        data = np.array(self.daily_returns_combined['pnl_absolute'].tolist())
        if days != -1:
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
        image_path = "heatmap.png"
        plt.savefig(image_path, bbox_inches='tight')
        st.write(hm)
        return image_path, box_width
        
        
    def avgReturns_merged(self, df1, df2):
        mean1, pct1 = avg_returns(df1)
        mean2, pct2 = avg_returns(df2)
        dt = df1.index[-1]
        cumsum = df1.loc[dt, 'cum_pnl']
        
        count_rows = len(df2[df2.index > dt])
        avg_returns = (mean1 * len(df1) + mean2 * len(df2) + cumsum * count_rows)/(len(df1) + len(df2)- len(df2[df2.index == dt]))
        avg_returns_pct = (pct1 * len(df1) + pct2 * len(df2) + cumsum/self.data1.initial_investment*100* count_rows)/(len(df1) + len(df2)- len(df2[df2.index == dt]))
    
        return round(avg_returns, 2), round(avg_returns_pct, 2)
    
    def freq_combined(self, df1, df2, r1, r2):
                
        if df1.index[-1] == df2.index[0]:
            ra = [df1['pnl_absolute'].iloc[-1] + df2['pnl_absolute'].iloc[0]]
            rs = np.array([df1['pnl_absolute'].iloc[-1], df2['pnl_absolute'].iloc[0]])
            r4 = self.data1.round_calc(ra)
            r5 = self.data1.round_calc(rs)
            return (r1 + r2 + r4- r5)
        else:
            return r1 + r2
        
    def combined_numLoss(self,df1, df2, r1, r2, i):
            if df1.index[-1] == df2.index[0]:
                ra = df1['pnl_absolute'].iloc[-1] + df2['pnl_absolute'].iloc[0]
                ra = 1 if ra == 0 else ra
                t = 1 if ra*i > 0 else 0
                rs = np.array([df1['pnl_absolute'].iloc[-1], df2['pnl_absolute'].iloc[0]])
                for i in rs:
                   k= np.where(np.sign(i) == 0, 1, np.sign(i))
                   if i*k > 0:
                       t-=1
                return (r1 + r2 + i*t)        
            else:
                return(r1 + r2)
            
    def avgProfit_merge(self, df1, df2,a1,a2, i):
        n1 = a1[1]
        n2 = a2[1]
        r1 = a1[0]
        r2 = a2[0]
        tot = r1 * n1 + r2 * n2
        if df1.index[-1] == df2.index[0]:
            sum = df1['pnl_absolute'].iloc[-1] + df2['pnl_absolute'].iloc[0]
            rs = np.array([df1['pnl_absolute'].iloc[-1], df2['pnl_absolute'].iloc[0]])
            if i == 1:
                if sum > 0:
                    elem = n1 + n2 -1
                    if rs[0] < 0:
                        tot += rs[0]
                        elem +=1
                    if rs[1] < 0:
                        tot += rs[1]
                        elem +=1
                    print(tot)
                    return (tot)/(elem) 
                else:
                    if rs[0] < 0 and rs[1] <0:
                        return tot/(n1 + n2)
                    else: 
                        positive_element = rs[0] if rs[0] > 0 else rs[1]
                        print(tot- positive_element)
                        return (tot - positive_element)/(n1 + n2-1)
            else:
                if sum < 0:
                    if rs[0] < 0 and rs[1] < 0:
                        return tot/(n1 + n2 -1)
                    else:
                        positive_element = rs[0] if rs[0] > 0 else rs[1]
                        return (tot + positive_element)/(n1 + n2)
                else:
                    if rs[0]  > 0 and rs[1] > 0:
                        return tot/(n1 + n2)
                    else: 
                        negative_element = rs[0] if rs[0] < 0 else rs[1]
                        return (tot - negative_element)/(n1 + n2-1)
        else:
            return tot/(n1 + n2)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
