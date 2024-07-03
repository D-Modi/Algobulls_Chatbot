
import sqlite3
from stratrgy_analysis import StatergyAnalysis  
import pickle
import pandas as pd
import numpy as np
from merge_csv import merge_csv
import re
import glob
import copy
import os
import shutil

#Extracts name of the Strategy from the given csv file name
def extract_text(filename):
    match = re.search(r'-(.*)\.csv$', filename)
    if match:
        return match.group(1)
    return filename

#Calculates and saves all the necessary data requiired in an array 
def calc(csv_name, is_dataframe=0, initial_inestent=150000, filename=None, save_data = False):
    
    #filename is not None when we pass a dataframe instead of  a csv file
    if filename is not None:
        strategy_code = extract_text(filename)
    else:
        strategy_code = extract_text(csv_name)
        
    Analysis = StatergyAnalysis(csv_name, is_dataframe=is_dataframe, number=initial_inestent)
    row= [strategy_code, Analysis.csv_data, Analysis.daily_returnts, Analysis.monthly_returns, Analysis.weekly_returns, Analysis.weekday_returns, Analysis.yearly_returns, Analysis.drawdown_max, Analysis.drawdown_pct,Analysis.avgProfit(Analysis.csv_data, -1),Analysis.avgProfit(Analysis.csv_data, 1), Analysis.profit[0], Analysis.loss[0], Analysis.short, Analysis.long, Analysis.avgTrades(Analysis.daily_returnts), Analysis.num_wins, Analysis.num_loss(Analysis.csv_data, -1), Analysis.Hit_Daywise, Analysis.roi(), Analysis.profit_factor, Analysis.pos, Analysis.neg, Analysis.yearlyVola(),Analysis.max_consecutive(Analysis.csv_data, 1), Analysis.max_consecutive(Analysis.csv_data, -1), Analysis.annual_std, Analysis.annual_mean, Analysis.initial_investment, Analysis.risk_free_rate, Analysis.equity_PctChange]
    
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

def append_sql(csv_name, is_dataframe=0, filename=None):
    head = ["Id","csv","Daily", "Monthly", "Weekly", "Weekday", "Yearly", 'MaxDD',"DDpct", "Aloss", "Awin", "MaxG", "MinG", "short", "long","Atr", "NumWins", "NumLoss", "HIT", "ROI", "PF", "Tot_pos","Tot_neg", "YVola", "MaxWinS", "MaxLossS","daily_annual_std", "daily_annual_mean","Inv","rfr", "EquityPct", "monthWR", "weekWR", "yearWR", "months6WR", "quarterWR", "Sharpe", "Calmar", "Sortino", "threeD", "thirtyD", 'twoW', "sixM","oneY", "twoY", "annual_mean", "annual_std", "disp_daily", "disp_monthly", "disp_weekly", "disp_weekday", "disp_yearly"]

    conn = sqlite3.connect('strategy_analysis.db')
    cursor = conn.cursor()
    
    row = calc(csv_name, is_dataframe=is_dataframe, filename=filename)
    print('APPEND')
    print("######################")
    print(row[0])
    stratergy = row[0]
    
    cursor.execute('SELECT * FROM StrategyData WHERE Id = ?', (row[0],))
    sql_data  = cursor.fetchone()
    sql_data = list(sql_data)
    
    if sql_data is not None:
        directory_name = f"files/{stratergy}"
        daily_returns_loaded = pd.read_pickle(f"{directory_name}/daily_returns.pkl")
        monthly_returns_loaded = pd.read_pickle(f"{directory_name}/monthly_returns.pkl")
        yearly_returns_loaded = pd.read_pickle(f"{directory_name}/yearly_returns.pkl")
        weekly_returns_loaded = pd.read_pickle(f"{directory_name}/weekly_returns.pkl")
        weekday_returns_loaded = pd.read_pickle(f"{directory_name}/weekday_returns.pkl")
        data = pd.read_pickle(f"{directory_name}/StrategyBacktestingPLBook-{stratergy}.pkl")
        q = [row[0], data, daily_returns_loaded, monthly_returns_loaded, weekly_returns_loaded, weekday_returns_loaded, yearly_returns_loaded]
        #Bytes to int
        pick = [2,3,4,5,6,9,10,19,30,31,32,33,34,35]
        for p in pick:
            sql_data[p] = pickle.loads(sql_data[p]) 
            
        if isinstance(sql_data[24], bytes):   
            sql_data[24] = int.from_bytes(sql_data[24], byteorder='little')      
        if isinstance(sql_data[25], bytes):   
            sql_data[25] = int.from_bytes(sql_data[25], byteorder='little')
        
        q.extend(sql_data[7:])
        q.extend(sql_data[2:7])
            
        merged = merge_csv(q, row)
        row_combined = [q[0], merged.csv_combined, merged.daily_returns_combined, merged.monthly_ret, merged.merged_df(q[4], row[4]), merged.merged_df(q[5], row[5]), merged.merged_df(q[6], row[6]), merged.drawdown_max, merged.drawdown_pct, merged.avgProfit_merge(q[1], row[1], q[9], row[9], -1), merged.avgProfit_merge(q[1], row[1], q[10], row[10], 1), max(q[11], row[11]), min(q[12], row[12]), merged.short, merged.long, merged.avgNumTrades,q[16] + row[16] , q[17]+row[17], merged.HIT, [merged.monthly_ret['cum_pnl'].iloc[-1], merged.monthly_ret['roi'].iloc[-1]] ,merged.ProfitFactor, q[21]+ row[21], q[22]+ row[22], merged.Yearly_Vola, merged.merged_max_cons(q[1], row[1], q[24], row[24], 1), merged.merged_max_cons(q[1], row[1], q[25], row[25], -1), merged.daily_combined_variance, merged.daily_combined_variance, q[28], q[29], merged.equity_PctChange]
        
        directory_path = f"files/{row[0]}"
        csv_name = f"StrategyBacktestingPLBook-{row[0]}.pkl"
        row_combined[2].to_pickle(f"{directory_path}/daily_returns.pkl")
        row_combined[6].to_pickle(f"{directory_path}/yearly_returns.pkl")    
        row_combined[3].to_pickle(f"{directory_path}/monthly_returns.pkl")
        row_combined[5].to_pickle(f"{directory_path}/weekday_returns.pkl")
        row_combined[4].to_pickle(f"{directory_path}/weekly_returns.pkl")
        row_combined[1].to_pickle(f"{directory_path}/{csv_name}")
        
        d = [30, 7, 365, 180, 120]
        for t in range(len(d)):
            start_date = merged.date_calc(returns=merged.daily_returns_combined, day=d[t])
            idx = merged.daily_returns_combined.index
            index_number = np.where(idx == start_date)[0][-1]
            num_rows = len(merged.daily_returns_combined)-index_number
            amt = merged.winR(q[2], row[2], q[31 + t], row[31 + t], num_rows*-1)
            row_combined.append([amt, num_rows])
        
        row_combined.extend([merged.sharpe_ratio, merged.Calmar_ratio, merged.sortino_ratio])
    
        T = [3,30,14,180,365,730]
        for a in T:
            row_combined.append(merged.Treturns(a, returns=merged.daily_returns_combined)[1])
        row_combined.extend([merged.combined_mean, merged.combined_variance])
       
        period = ['Day', 'Month', 'Week', 'WeekDay', 'Year']
        for i in range(len(period)):
            df1 = q[i+2]
            df2 = row[i+2]
            a = q[47+i]
            b = row[47+i]
            period_wise_returns = []
            
            if i !=3:
                period_wise_returns.extend([period[i], merged.avgReturns_merged(a[1], b[1], df1, df2), merged.freq_combined(df1, df2, a[2], b[2]), merged.comb_tradingNum(df1, df2, a[3], b[3]), merged.combined_numLoss(df1, df2, a[4], b[4], 1), merged.combined_numLoss(df1, df2, a[5], b[5], -1)])
                if i ==0:
                    period_wise_returns.extend([max([a[6], b[6]], key=lambda lst: lst[0]), min([a[7], a[7]], key=lambda lst: lst[0]), merged.merged_max_cons(df1, df2, a[8], b[8], 1), merged.merged_max_cons(df1, df2, a[9], b[9], -1)])
                else:
                    period_wise_returns.extend([merged.max_min_Profit(1, df1, df2, a[6], b[6], row_combined[i+2]), merged.max_min_Profit(-1, df1, df2, a[7], b[7], row_combined[i+2]) ])
            if i ==3: 
                period_wise_returns.extend([merged.max_profit(row_combined[5], 1), merged.max_profit(row_combined[5], -1)])
            row_combined[i+2] = (period_wise_returns)    
        #Replacing the csv file with a 1 (to preserve the index)
        row_combined[1] = 1
                             
        for p in pick:
            row_combined[p] = pickle.dumps(row_combined[p])
            
        cursor.execute('''REPLACE INTO StrategyData VALUES (?, ?, ?, ?, ?,?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row_combined)
    conn.commit()
    conn.close()
    
def insert_sql(csv_name, is_dataframe=0, filename=None):

    conn = sqlite3.connect('strategy_analysis.db')
    cursor = conn.cursor()
    print("INSERT")
    print(filename)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    row_pickled = calc(csv_name, is_dataframe=is_dataframe, filename=filename)  
    strategy_code = row_pickled[0] 
    directory_path = f"files/{strategy_code}"
    csv_name = f"StrategyBacktestingPLBook-{strategy_code}.pkl"
    
    if not os.path.isdir(directory_path):
        os.makedirs(f"files/{strategy_code}")
        
    row_pickled[2].to_pickle(f"{directory_path}/daily_returns.pkl")
    row_pickled[6].to_pickle(f"{directory_path}/yearly_returns.pkl")    
    row_pickled[3].to_pickle(f"{directory_path}/monthly_returns.pkl")
    row_pickled[5].to_pickle(f"{directory_path}/weekday_returns.pkl")
    row_pickled[4].to_pickle(f"{directory_path}/weekly_returns.pkl")
    row_pickled[1].to_pickle(f"{directory_path}/{csv_name}")
    
    replace = [2,3,4,5,6]
    for i in replace:
        row_pickled[i] = row_pickled[i+45]
    row_pickled[1] = 1
    row_pickled = row_pickled[:-5]
    print(len(row_pickled))
    pick = [2,3,4,5,6,9,10,19,30,31,32,33,34,35]
    for p in pick:
        row_pickled[p] = pickle.dumps(row_pickled[p])
        
    cursor.execute('''INSERT OR REPLACE INTO StrategyData VALUES (?, ?, ?,?, ?,?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row_pickled)
    
    conn.commit()
    conn.close()


def delete_id(id_to_delete):
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect('strategy_analysis.db')
        cursor = conn.cursor()

        # Execute the DELETE statement to remove the row with the specified ID
        cursor.execute('DELETE FROM StrategyData WHERE Id = ?', (id_to_delete,))

        # Commit the transaction to save the changes
        conn.commit()

        print(f"##############################################Row with ID '{id_to_delete}' has been deleted successfully################################################.")
    except sqlite3.Error as e:
        print("SQLite error:", e)
    
    conn.close()

def run():
    conn = sqlite3.connect('strategy_analysis.db')
    cursor = conn.cursor()

    cursor.execute('''DROP TABLE IF EXISTS StrategyData''')
    # # # Create a table
    cursor.execute('''CREATE TABLE IF NOT EXISTS StrategyData (
        Id TEXT PRIMARY KEY,
        Csv INTEGER,
        Daily BLOB,
        Monthly BLOB,
        Weekly BLOB,
        Weekday BLOB,
        Yearly BLOB,
        MaxDD REAL,
        DDpct REAL,
        Aloss BLOB,
        Awin BLOB,
        MaxG REAL,
        MinG REAL,
        short INTEGER,
        long INTEGER,
        Atr REAL,
        NumWins INTEGER,
        NumLoss INTEGER,
        HIT REAL,
        ROI BLOB,
        PF REAL,
        Tot_pos REAL,
        Tot_neg REAL,
        YVola REAL,
        MaxWinS INTEGER,
        MaxLossS INTEGER,
        daily_annual_std REAL,
        daily_annual_mean REAL,
        Inv REAL,
        RFR REAL,
        EquityPct BLOB,
        monthWR BLOB,
        weekWR BLOB,
        yearWR BLOB,
        months6WR BLOB,
        quarterWR BLOB,
        Sharpe REAL,
        Calmar REAL,
        Sortino REAL,
        threeD REAL,
        thirtyD REAL,
        twoW REAL,
        sixM REAL,
        OneY REAL,
        twoY REAL,
        annual_std REAL,
        annual_mean REAL
    )''')

    conn.commit()
    conn.close()

    #  Column names of sql  database
    # # daily = ["period", "avgRet", "avgRetPct", "freq", "numD", "numProfit", "numLoss", "MPDay", "MaxG", "LPDay", "MinG", "MaxWinS", "MaxLossS"]

    path = "files/StrategyBacktestingPLBook-*.csv"
    Files = []

    for file in glob.glob(path, recursive=True):
        Files.append(file)
    #Files = ["files/StrategyBacktestingPLBook-STAB679.csv"]
    for i in Files:
        insert_sql(i)
            
