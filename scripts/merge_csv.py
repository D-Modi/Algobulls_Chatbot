# consecutoive days, max_profit, htmap, monthly returns and moi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import streamlit as st
import seaborn as sn
from  matplotlib.colors import LinearSegmentedColormap
from statistics import mean 
from stratrgy_analysis import StatergyAnalysis
from datetime import datetime
from dateutil.relativedelta import relativedelta

class merge_csv:
    
    def __init__(self, csv1, csv2):
        self.data1 = csv1
        self.data2 = csv2
        self.initial_investment = self.data1[28]
        self.data1_csv = self.data1[1]
        self.data2_csv = self.data2[1]
        # self.daily_returns_combined = pd.concat([self.data1[2], self.data2[2]]) # Only for "pnl_absolute"; not for cum_pnl
        # self.daily_returns_combined['cum_pnl'] = self.daily_returns_combined['pnl_absolute'].cumsum()
        self.daily_returns_combined = self.merge_day(self.data1[2], self.data2[2])
        self.daily_combined_mean = (self.data1[27] * len(self.data1[2])+ self.data2[27] * len(self.data2[2]) )/ (len(self.data1[2])+ len(self.data2[2]))
        self.daily_combined_variance = (((len(self.data1[2])-1) * self.data1[26]**2 + (len(self.data2[2])-1) * self.data2[26] **2) +  len(self.data1[2]) * (self.data1[27] - self.daily_combined_mean)**2 +  len(self.data2[2]) * (self.data2[27] - self.daily_combined_mean)**2) / (len(self.data1[2])+ len(self.data2[2])-2)
        self.daily_combined_variance = self.daily_combined_variance ** 0.5
        self.combined_mean = (self.data1[45] * len(self.data1[1])+ self.data2[45] * len(self.data2[1]) )/ (len(self.data1[1])+ len(self.data2[1]))
        self.combined_variance = (((len(self.data1[1])-1) * self.data1[46]**2 + (len(self.data2[1])-1) * self.data2[46] **2) +  len(self.data1[1]) * (self.data1[45] - self.combined_mean)**2 +  len(self.data2[1]) * (self.data2[45] - self.combined_mean)**2) / (len(self.data1[1])+ len(self.data2[1])-2)
        self.combined_variance = self.combined_variance ** 0.5
        self.risk_free_rate = self.data1[29]
        self.Yearly_Vola = round(self.combined_variance *100, 2)
        self.sharpe_ratio = round((self.daily_combined_mean *np.sqrt(252) - self.risk_free_rate)/ self.daily_combined_variance, 2)
        #self.min_DDPct = min(self.data1[8], self.data2[8])
        #self.equity = pd.concat([self.data1.equity, self.data2.equity])
        transition_equity_change = (self.data2_csv['equity_curve'].iloc[0] - self.data1_csv['equity_curve'].iloc[-1]) / self.data1_csv['equity_curve'].iloc[-1]
        # self.equity_PctChange = self.data1.equity_PctChange.copy()
        # self.equity_PctChange.loc[len(self.equity_PctChange)] = transition_equity_change
        # self.equity_PctChange = self.equity_PctChange.append(self.data2.equity_PctChange, ignore_index=True)
        self.equity_PctChange = self.data1[30].copy()
        self.equity_PctChange = pd.concat([self.equity_PctChange, pd.Series([transition_equity_change])], ignore_index=True)
        self.equity_PctChange = pd.concat([self.equity_PctChange, self.data2[30]], ignore_index=True)
        self.sortino_ratio = self.Sortino_Ratio()
        self.drawdown_data2(max_eq=self.data1_csv['equity_cum_max'].iloc[-1])
        self.drawdown_column = pd.concat([self.data1_csv[['drawdown', 'drawdown_pct']], self.data2_csv[['drawdown', 'drawdown_pct']]])
        self.drawdown_max = round(self.drawdown_column['drawdown'].min(), 2)
        self.drawdown_pct = round(self.drawdown_column['drawdown_pct'].min(), 2)
        self.Calmar_ratio = round(self.daily_combined_mean*np.sqrt(252) / self.drawdown_pct * -100, 2)
        self.HIT = round((self.data1[47][4] + self.data2[47][4])/(self.data1[47][3]+ self.data2[47][3])* 100, 2)
        self.long = self.data1[14] + self.data2[14]
        self.short = self.data1[13] +self.data2[13]
        self.avgNumTrades = round((len(self.data1[1])+ len(self.data2[1]))/(len(self.data1[2]) + len(self.data2[2])), 2)
        self.ProfitFactor = round((self.data1[21] + self.data2[21])/(self.data1[22] + self.data2[22])* -1 , 2) 
        #self.maxProfit = max([self.data1.profit, self.data2.profit], key=lambda lst: lst[0])
        #self.maxLoss = min([self.data1.loss, self.data2.loss], key=lambda lst: lst[0])  
        self.monthly_ret = self.merged_df(self.data1[3], self.data2[3])
        self.csv_combined = pd.concat([self.data1_csv, self.data2_csv])
        
    def merged_df(self, df1, df2):
        dt = df1.index[-1]
        result_df = df1.add(df2, fill_value=0)
      
        if 'cum_pnl' in result_df.columns: 
            cumsum = df1['cum_pnl'].iloc[-1]  
            result_df.loc[result_df.index > dt, 'cum_pnl'] += cumsum
        if 'roi' in result_df.columns: 
            roi_t = df1['roi'].iloc[-1]  
            result_df.loc[result_df.index > dt, 'roi'] += roi_t
            
        return result_df
    
    def merged_monthly(self, df1, df2):
        if df1.index[-1] != df2.index[0]:
            if 'cum_pnl' in df1.columns:
                df2['cum_pnl'].iloc[0] += df1['cum_pnl'].iloc[-1]
            if 'roi' in df1.columns:
                df2['roi'].iloc[0] += df1['roi'].iloc[-1]
            result_df = pd.concat([df1, df2])

        return result_df
    
    def merge_day(self, df1, df2):
        result_df = pd.concat([df1, df2])
        if 'cum_pnl' in result_df.columns:
            cumsum = df1['cum_pnl'].iloc[-1]   
            result_df.loc[result_df.index > df1.index[-1], 'cum_pnl'] += cumsum
        if 'roi' in result_df.columns: 
            roi_t = df1['roi'].iloc[-1]  
            result_df.loc[result_df.index > df1.index[-1], 'roi'] += roi_t
        return result_df   
     
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

    def max_min_Profit(self, i, returns1,returns2, MaxProfits1, MaxProfits2, df):
        #MaxProfits1 = self.data1.max_profit(returns1, i)
        #MaxProfits2 = self.data1.max_profit(returns2, i)
        
        if i == 1:
            MP_temp = max([MaxProfits1, MaxProfits2], key=lambda lst: lst[0])
            if returns1.index[-1] == returns2.index[0]:
                MaxProfit3  = [returns1['pnl_absolute'].iloc[-1] + returns2['pnl_absolute'].iloc[0], returns1.index[-1]]
                MP = max(MaxProfits1[0], MaxProfits2[0], MaxProfit3[0])
                if MP == MaxProfit3[0]:
                    return MaxProfit3
                if MP == MaxProfits1[0]:
                    if MaxProfits1[1] != returns1.index[-1]:
                        return MaxProfits1   
                if MP == MaxProfits2[0]:
                    if MaxProfits2[1] != returns1.index[-1]:
                        return MaxProfits2  
                return self.max_profit(df, i)              
            else:
                return MP_temp
        else:
            MP_temp = min([MaxProfits1, MaxProfits2], key=lambda lst: lst[0])
            if returns1.index[-1] == returns2.index[0]:
                MaxProfit3  = returns1['pnl_absolute'].iloc[-1] + returns2['pnl_absolute'].iloc[0]
                MP = min(MaxProfits1[0], MaxProfits2[0], MaxProfit3)
                if MP == MaxProfit3:
                    return MP, returns1.index[-1]
                if MP == MaxProfits1[0]:
                    if MaxProfits1[1] != returns1.index[-1]:
                        return MaxProfits1   
                if MP == MaxProfits2[0]:
                    if MaxProfits2[1] != returns1.index[-1]:
                        return MaxProfits2  
                return self.max_profit(df, i)              
            else:
                return MP_temp  


    def Sortino_Ratio(self):
        downside_returns = np.where(self.equity_PctChange < 0, self.equity_PctChange, 0)
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return round((self.daily_combined_mean*np.sqrt(252) - self.risk_free_rate) / downside_deviation, 2)
    
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
    
    def date_calc(self, day=0, returns=None):
        if returns is None:
            returns = self.daily_returns_combined
            
        last_date = datetime.strptime(returns.index[-1], '%Y-%m-%d')
        start_date = last_date - relativedelta(days=day)
        
        for i in returns.index:
            date =  datetime.strptime(i, '%Y-%m-%d')
            if date >= start_date:
                start_date = date
                break
        new_date_str = start_date.strftime('%Y-%m-%d')
        return new_date_str
        
    def Treturns(self, days, returns=None):
        if returns is None:
            returns = self.daily_returns_combined
            
        new_date_str = self.date_calc(day=days, returns=returns)
        print(new_date_str)
        final_equity_value = returns['cum_pnl'].iloc[-1]
        start_equity_value = returns.loc[new_date_str, 'cum_pnl']
        gain = final_equity_value - start_equity_value
        return gain, round(gain*100/self.initial_investment, 2)
        
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
        
        
    def avgReturns_merged(self, avgR1, avgR2, df1, df2):
        mean1, pct1 = avgR1[0], avgR1[1]
        mean2, pct2 = avgR2[0], avgR2[1]
        dt = df1.index[-1]
        cumsum = df1.loc[dt, 'cum_pnl']
        
        count_rows = len(df2[df2.index > dt])
        avg_returns = (mean1 * len(df1) + mean2 * len(df2) + cumsum * count_rows)/(len(df1) + len(df2)- len(df2[df2.index == dt]))
        avg_returns_pct = (pct1 * len(df1) + pct2 * len(df2) + cumsum/self.initial_investment*100* count_rows)/(len(df1) + len(df2)- len(df2[df2.index == dt]))
    
        return [round(avg_returns, 2), round(avg_returns_pct, 2)]
    
    def round_calc(self, retrurn):
        profit = np.zeros(12)
        for r in retrurn:
            val = int(r // 1000 + 6)
            #print(val)
            if val < 0:
                val = 0
            if val> 11:
                val = 11
            profit[val] +=1
        return profit
    
    def comb_tradingNum(self, df1, df2, r1, r2):
        if df1.index[-1] == df2.index[0]:
            return r1 + r2 -1
        return r1 + r2
    
    def freq_combined(self, df1, df2, r1, r2):
                
        if df1.index[-1] == df2.index[0]:
            ra = [df1['pnl_absolute'].iloc[-1] + df2['pnl_absolute'].iloc[0]]
            rs = np.array([df1['pnl_absolute'].iloc[-1], df2['pnl_absolute'].iloc[0]])
            r4 = self.round_calc(ra)
            r5 = self.round_calc(rs)
            return (r1 + r2 + r4- r5)
        else:
            return r1 + r2
        
    def combined_numLoss(self,df1, df2, r1, r2, i):
            
            if df1.index[-1] == df2.index[0]:
                ra = df1['pnl_absolute'].iloc[-1] + df2['pnl_absolute'].iloc[0]
                ra = 1 if ra == 0 else ra
                t = 1 if ra*i > 0 else 0
                rs = np.array([df1['pnl_absolute'].iloc[-1], df2['pnl_absolute'].iloc[0]])
                for j in rs:
                   k= np.where(np.sign(j) == 0, 1, np.sign(j))
                   if i*k > 0:
                       t-=1
                #print(r1, r2, t, i,  r1 + r2 + i*t) 
                return (r1 + r2 + i*t)        
            else:
                #print(r1, r2,i, r1 + r2)
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
                    return round((tot)/(elem), 2) , elem
                else:
                    if rs[0] < 0 and rs[1] <0:
                        return round(tot/(n1 + n2), 2), n1 + n2
                    else: 
                        positive_element = rs[0] if rs[0] > 0 else rs[1]
                        print(tot- positive_element)
                        return round((tot - positive_element)/(n1 + n2-1), 2) , n1 + n2 -1
            else:
                if sum < 0:
                    if rs[0] < 0 and rs[1] < 0:
                        return round(tot/(n1 + n2 -1), 2), n1 + n2 -1
                    else:
                        positive_element = rs[0] if rs[0] > 0 else rs[1]
                        return round((tot + positive_element)/(n1 + n2), 2), n1 + n2
                else:
                    if rs[0]  > 0 and rs[1] > 0:
                        return round(tot/(n1 + n2), 2), n1 + n2
                    else: 
                        negative_element = rs[0] if rs[0] < 0 else rs[1]
                        return round((tot - negative_element)/(n1 + n2-1), 2), n1+n2-1
        else:
            return round(tot/(n1 + n2), 2), n1+n2
        
    def merged_max_cons(self, df1, df2, r1, r2, i):
        if i == 1:
            data = df2[df2['pnl_absolute'] < 0]
            if len(data) > 0:
                first_negative_index = df2[df2['pnl_absolute'] < 0].index[0]
                idx = df2.index
                first_negative_row = np.where(idx == first_negative_index)[0][0]   
            else:
                first_negative_row = len(df2)

            # Identify the index of the last negative value in 'pnl'
            data = df1[df1['pnl_absolute'] < 0] 
            if len(data) > 0:
                last_negative_index = df1[df1['pnl_absolute'] < 0].index[-1]
                idx = df1.index
                last_negative_row = np.where(idx == last_negative_index)[0][-1]
            else:
                last_negative_row = -1
        else:
            first_negative_index = df2[df2['pnl_absolute'] > 0].index[0]
            idx = df2.index
            first_negative_row = np.where(idx == first_negative_index)[0][0]
            
            last_negative_index = df1[df1['pnl_absolute'] > 0].index[-1]
            idx = df1.index
            last_negative_row = np.where(idx == last_negative_index)[0][-1]
            
        positive_rows_from_start = first_negative_row 
        positive_rows_from_end = len(df1) - (last_negative_row + 1)
        total_positive_rows = positive_rows_from_end + positive_rows_from_start

    
        return  max(r1, r2, total_positive_rows)
    
    def winR(self, df1, df2, r1, r2, d):
        print(len(df2), d)
        if len(df2)>=d*-1:
            return r2[0]
        rem = d*-1 - len(df2)
        index_pos = -1*r1[1] + len(df1)
        if rem < r1[1]:
            per = df1.iloc[-1*r1[1]:].head(rem)
            wins = len(per[per['pnl_absolute']>0])
            fin = (r1[0] * r1[1] + r2[0] * len(df2) - wins)/d*-1
        else:
            diff = rem-r1[1]
            rows_above = df1.iloc[max(index_pos-diff, 0):index_pos]
            wins = len(rows_above[rows_above['pnl_absolute']>0])
            fin = (r1[0] * r1[1] + r2[0] * r2[1] + wins)/d*-1
        return fin
          
    def drawdown_data2(self, max_eq=0 ):

        self.data2_csv['cum_max'] = self.data2_csv['equity_curve'].cummax()
        self.data2_csv.loc[self.data2_csv['cum_max'] <= max_eq, 'cum_max'] = max_eq  
        #self.data2_csv['drawdown'] = self.data2_csv['equity_pnl'] - self.data2_csv['cum_max']        
        self.data2_csv['drawdown'] = self.data2_csv['equity_curve'] - self.data2_csv['cum_max']
        self.data2_csv['drawdown_pct'] = self.data2_csv['drawdown_percentage']
        
    def Treturns(self,t, returns=None):
        if returns is None:
            returns = self.daily_returnts
            
        cum_pnl = returns['cum_pnl'].tolist()
        cum_pnl = cum_pnl[-1*t:]
        ret = cum_pnl[-1] - cum_pnl[0]
        return ret, round(ret*100/self.initial_investment, 2)
            
