
from stratrgy_analysis import StatergyAnalysis  
import re

#Extracts name of the Strategy from the given csv file name
def extract_text(filename):
    match = re.search(r'-(.*)\.csv$', filename)
    if match:
        return match.group(1)
    return filename

#Calculates and saves all the necessary data requiired in an array 
def calc(csv_name, is_dataframe=0, initial_investment=150000, filename=None):
    
    #filename is not None when we pass a dataframe instead of  a csv file
    if filename is not None:
        stn = extract_text(filename)
    else:
        stn = extract_text(csv_name)
        
    Analysis = StatergyAnalysis(csv_name, is_dataframe=is_dataframe, number=initial_investment)
    row= [stn, Analysis.csv_data, Analysis.daily_returnts, Analysis.monthly_returns, Analysis.weekly_returns, Analysis.weekday_returns, Analysis.yearly_returns, Analysis.drawdown_max, Analysis.drawdown_pct,Analysis.avgProfit(Analysis.csv_data, -1),Analysis.avgProfit(Analysis.csv_data, 1), Analysis.profit[0], Analysis.loss[0], Analysis.short, Analysis.long, Analysis.avgTrades(Analysis.daily_returnts), Analysis.num_wins, Analysis.num_loss(Analysis.csv_data, -1), Analysis.hit, Analysis.roi(), Analysis.profit_factor, Analysis.pos, Analysis.neg, Analysis.yearlyVola(),Analysis.max_consecutive(Analysis.csv_data, 1), Analysis.max_consecutive(Analysis.csv_data, -1), Analysis.annual_std, Analysis.annual_mean, Analysis.initial_investment, Analysis.risk_free_rate, Analysis.equity_PctChange]

    #Win_Rate by Period
    d = [-21, -6, -213, -101, -59]
    for t in d:
        if len(Analysis.daily_returnts) > -1*t:
            last_month_data = Analysis.daily_returnts.iloc[t:]
        else:
            last_month_data = Analysis.daily_returnts
        row.append(Analysis.win_rate(last_month_data))
    
    row.extend([Analysis.Sharpe(), Analysis.Calmar(), Analysis.Sortino()])
    T = [4,22,11,101,252,504]
    for a in T:
        row.append(Analysis.Treturns(a)[1])
        
    row.extend([Analysis.annual_mean, Analysis.annual_std])

    #Period wise returns. We are calculating max_consecutive streak only for daily analysis. And for weekday wise analysis, we just return the max/min profits and max/min proftable days
    period = ['Day', 'Month', 'Week', 'WeekDay', 'Year']
    for i in range(len(period)):
        daily_returns = row[i+2]
        period_wise_returns = []
        if i != 3:
            period_wise_returns.extend([period[i], Analysis.avgReturns(daily_returns), Analysis.round_calc(daily_returns['pnl_absolute']), len(daily_returns), Analysis.num_loss(daily_returns, 1), Analysis.num_loss(daily_returns, -1), Analysis.max_profit(daily_returns, 1), Analysis.max_profit(daily_returns, -1)])
        if i == 0:
            period_wise_returns.extend([Analysis.max_consecutive(daily_returns, 1), Analysis.max_consecutive(daily_returns, -1)])
        if i == 3:
            period_wise_returns = [Analysis.max_profit(daily_returns, 1), Analysis.max_profit(daily_returns, -1) ]
        row.append(period_wise_returns)

    return row
    
