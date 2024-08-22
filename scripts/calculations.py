
from stratrgy_analysis import StatergyAnalysis  
import re

#Extracts name of the Strategy from the given csv file name
def extract_text(filename):
    match = re.search(r'-(.*)\.csv$', filename)
    if match:
        return match.group(1)
    return filename

#Calculates and saves all the necessary data requiired in an array 
def calc(csv_name, is_dataframe=0, initial_investment=150000, filename=None, save_data = False):
    
    #filename is not None when we pass a dataframe instead of  a csv file
    if filename is not None:
        strategy_code = extract_text(filename)
    else:
        strategy_code = extract_text(csv_name)
        
    Analysis = StatergyAnalysis(csv_name, is_dataframe=is_dataframe, number=initial_investment)
    row= [strategy_code, Analysis.csv_data, Analysis.daily_returnts, Analysis.monthly_returns, Analysis.weekly_returns, Analysis.weekday_returns, Analysis.yearly_returns, Analysis.drawdown_max, Analysis.drawdown_pct,Analysis.avgProfit(Analysis.csv_data, -1),Analysis.avgProfit(Analysis.csv_data, 1), Analysis.profit[0], Analysis.loss[0], Analysis.short, Analysis.long, Analysis.avgTrades(Analysis.daily_returnts), Analysis.num_wins, Analysis.num_loss(Analysis.csv_data, -1), Analysis.hit, Analysis.roi(), Analysis.profit_factor, Analysis.pos, Analysis.neg, Analysis.yearlyVola(),Analysis.max_consecutive(Analysis.csv_data, 1), Analysis.max_consecutive(Analysis.csv_data, -1), Analysis.annual_std, Analysis.annual_mean, Analysis.initial_investment, Analysis.risk_free_rate, Analysis.equity_PctChange]
    
    #Win_Rate by Period
    d = [30, 7, 365, 180, 90]
    for t in d:
        start_date = Analysis.date_calc(day=t)
        index_number = Analysis.daily_returnts.index.get_loc(start_date)
        num_rows = len(Analysis.daily_returnts)-index_number
        last_month_data = Analysis.daily_returnts.iloc[index_number:]
    
        row.append([Analysis.win_rate(last_month_data), num_rows])
    
    row.extend([Analysis.Sharpe(), Analysis.Calmar(), Analysis.Sortino()])
    T = [3,30,14,180,365,730]
    for a in T:
        row.append(Analysis.Treturns(day=a)[1])
        
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


def calc_for_customer_plb(strategy_code, Analysis):
    
    #filename is not None when we pass a dataframe instead of  a csv file
        
    row= [strategy_code, Analysis.csv_data, Analysis.daily_returnts, Analysis.monthly_returns, Analysis.weekly_returns, Analysis.weekday_returns, Analysis.yearly_returns, Analysis.drawdown_max, Analysis.drawdown_pct,Analysis.avgProfit(Analysis.csv_data, -1),Analysis.avgProfit(Analysis.csv_data, 1), Analysis.profit[0], Analysis.loss[0], Analysis.short, Analysis.long, Analysis.avgTrades(Analysis.daily_returnts), Analysis.num_wins, Analysis.num_loss(Analysis.csv_data, -1), Analysis.hit, Analysis.roi(), Analysis.profit_factor, Analysis.pos, Analysis.neg, Analysis.yearlyVola(),Analysis.max_consecutive(Analysis.csv_data, 1), Analysis.max_consecutive(Analysis.csv_data, -1), Analysis.annual_std, Analysis.annual_mean, Analysis.initial_investment, Analysis.risk_free_rate, Analysis.equity_PctChange]
    
    #Win_Rate by Period
    d = [30, 7, 365, 180, 120]
    for t in d:
        start_date = Analysis.date_calc(day=t)
        index_number = Analysis.daily_returnts.index.get_loc(start_date)
        num_rows = len(Analysis.daily_returnts)-index_number
        last_month_data = Analysis.daily_returnts.iloc[index_number:]
    
        row.append([Analysis.win_rate(last_month_data), num_rows])
    
    row.extend([Analysis.Sharpe(), Analysis.Calmar(), Analysis.Sortino()])
    T = [3,30,14,180,365,730]
    for a in T:
        row.append(Analysis.Treturns(day=a)[1])
        
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
