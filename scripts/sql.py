
import sqlite3
from stratrgy_analysis import StatergyAnalysis  # Assuming you have a file named strategy_analysis.py
import pickle
import pandas as pd
import numpy as np
from trial import merge_csv
import re
import glob
import copy

def extract_text(filename):
    match = re.search(r'-(.*)\.csv$', filename)
    if match:
        return match.group(1)
    return None

def calc(csv_name, i=0, initial_inestent=150000, filename=None):
    if filename is not None:
        stn = filename
    else:
        stn = extract_text(csv_name)
    Ana = StatergyAnalysis(csv_name, i=i, number=initial_inestent)
    row= [stn, Ana.csv_data, Ana.daily_returnts, Ana.monthly_returns, Ana.weekly_returns, Ana.weekday_returns, Ana.yearly_returns, Ana.drawdown_max, Ana.drawdown_pct,Ana.avgProfit(Ana.csv_data, -1),Ana.avgProfit(Ana.csv_data, 1), Ana.profit[0], Ana.loss[0], Ana.short, Ana.long, Ana.avgTrades(Ana.daily_returnts), Ana.num_wins, Ana.num_loss(Ana.csv_data, -1), Ana.hit, Ana.roi(), Ana.profit_factor, Ana.pos, Ana.neg, Ana.yearlyVola(),Ana.max_consecutive(Ana.csv_data, 1), Ana.max_consecutive(Ana.csv_data, -1), Ana.annual_std, Ana.annual_mean, Ana.initial_investment, Ana.risk_free_rate, Ana.equity_PctChange]

    d = [-21, -6, -213, -101, -59]
    for t in d:
        if len(Ana.daily_returnts) > -1*t:
            last_month_data = Ana.daily_returnts.iloc[t:]
        else:
            last_month_data = Ana.daily_returnts
        row.append(Ana.win_rate(last_month_data))
    
    row.extend([Ana.Sharpe(), Ana.Calmar(), Ana.Sortino()])

    T = [4,22,11,101,252,504]
    for a in T:
        row.append(Ana.Treturns(a)[1])
        
    row.extend([Ana.annual_mean, Ana.annual_std])

    period = ['Day', 'Month', 'Week', 'WeekDay', 'Year']
    for i in range(len(period)):
        daily_returns = row[i+2]
        damn = []
        if i != 3:
            damn.extend([period[i], Ana.avgReturns(daily_returns), Ana.round_calc(daily_returns['pnl_absolute']), len(daily_returns), Ana.num_loss(daily_returns, 1), Ana.num_loss(daily_returns, -1), Ana.max_profit(daily_returns, 1), Ana.max_profit(daily_returns, -1)])
        if i == 0:
            damn.extend([Ana.max_consecutive(daily_returns, 1), Ana.max_consecutive(daily_returns, -1)])
        if i == 3:
            damn = [Ana.max_profit(daily_returns, 1), Ana.max_profit(daily_returns, -1) ]
        row.append(damn)

    return row
    
def insert_sql(csv_name, i=0, filename=None):

    conn = sqlite3.connect('strategy_analysis.db')
    cursor = conn.cursor()
    if filename is not None:
        stn = extract_text(filename)
    else: 
        stn = extract_text(csv_name)
    Ana = StatergyAnalysis(csv_name, i)
    row= [stn, Ana.csv_data, Ana.daily_returnts, Ana.monthly_returns, Ana.weekly_returns, Ana.weekday_returns, Ana.yearly_returns, Ana.drawdown_max, Ana.drawdown_pct,Ana.avgProfit(Ana.csv_data, -1),Ana.avgProfit(Ana.csv_data, 1), Ana.profit[0], Ana.loss[0], Ana.short, Ana.long, Ana.avgTrades(Ana.daily_returnts), Ana.num_wins, Ana.num_loss(Ana.csv_data, -1), Ana.hit, Ana.roi(), Ana.profit_factor, Ana.pos, Ana.neg, Ana.yearlyVola(),Ana.max_consecutive(Ana.csv_data, 1), Ana.max_consecutive(Ana.csv_data, -1), Ana.annual_std, Ana.annual_mean, Ana.initial_investment, Ana.risk_free_rate, Ana.equity_PctChange]

    d = [-21, -6, -213, -101, -59]
    for t in d:
        if len(Ana.daily_returnts) > -1*t:
            last_month_data = Ana.daily_returnts.iloc[t:]
        else:
            last_month_data = Ana.daily_returnts
        row.append(Ana.win_rate(last_month_data))
    
    row.extend([Ana.Sharpe(), Ana.Calmar(), Ana.Sortino()])
    T = [4,22,11,101,252,504]
    for a in T:
        
        row.append(Ana.Treturns(a)[1])
        
    row.extend([Ana.annual_mean, Ana.annual_std])

    period = ['Day', 'Month', 'Week', 'WeekDay', 'Year']
    for i in range(len(period)):
        daily_returns = row[i+2]
        damn = []
        if i != 3:
            damn.extend([period[i], Ana.avgReturns(daily_returns), Ana.round_calc(daily_returns['pnl_absolute']), len(daily_returns), Ana.num_loss(daily_returns, 1), Ana.num_loss(daily_returns, -1), Ana.max_profit(daily_returns, 1), Ana.max_profit(daily_returns, -1)])
        if i == 0:
            damn.extend([Ana.max_consecutive(daily_returns, 1), Ana.max_consecutive(daily_returns, -1)])
        if i == 3:
            damn = [Ana.max_profit(daily_returns, 1), Ana.max_profit(daily_returns, -1) ]
        row.append(damn)
       
    row_pickled =  copy.deepcopy(row)   
    pick = [1,2,3,4,5,6,9,10,19,30,47,48,49,50,51]
    for p in pick:
            row_pickled[p] = pickle.dumps(row_pickled[p])

    
    cursor.execute('SELECT * FROM StrategyData WHERE Id = ?', (stn,))
    q  = cursor.fetchone()

    if q is not None:
        pick = [1,2,3,4,5,6,9,10,19,30,47,48,49,50,51]
        q = list(q)
    
        for p in pick:
            q[p] = pickle.loads(q[p])
            
        if isinstance(q[24], bytes):   
            q[24] = int.from_bytes(q[24], byteorder='little')      
        if isinstance(q[25], bytes):   
            q[25] = int.from_bytes(q[25], byteorder='little')
            
        merged = merge_csv(q, row)

        row_combined = [q[0], merged.csv_combined, merged.daily_returns_combined, merged.monthly_ret, merged.merged_df(q[4], row[4]), merged.merged_df(q[5], row[5]), merged.merged_df(q[6], row[6]), merged.drawdown_max, merged.drawdown_pct, merged.avgProfit_merge(q[1], row[1], q[9], row[9], -1), merged.avgProfit_merge(q[1], row[1], q[10], row[10], 1), max(q[11], row[11]), min(q[12], row[12]), merged.short, merged.long, merged.avgNumTrades,q[16] + row[16] , q[17]+row[17], merged.HIT, [merged.monthly_ret['cum_pnl'].iloc[-1], merged.monthly_ret['roi'].iloc[-1]] ,merged.ProfitFactor, q[21]+ row[21], q[22]+ row[22], merged.Yearly_Vola, merged.merged_max_cons(q[1], row[1], q[24], row[24], 1), merged.merged_max_cons(q[1], row[1], q[25], row[25], -1), merged.daily_combined_variance, merged.daily_combined_variance, q[28], q[29], merged.equity_PctChange]
    
        d = [-21, -6, -213, -101, -59]
        for t in range(len(d)):
            amt = merged.winR(q[2], row[2], q[31 + t], row[31 + t], d[t])
            row_combined.append(amt)
        
        row_combined.extend([merged.sharpe_ratio, merged.Calmar_ratio, merged.sortino_ratio])
    
        T = [4,22,11,101,252,504]
        for a in T:
            row_combined.append(Ana.Treturns(a, merged.daily_returns_combined)[1])
        row_combined.extend([merged.combined_mean, merged.combined_variance])
    
        period = ['Day', 'Month', 'Week', 'WeekDay', 'Year']
        for i in range(len(period)):
            df1 = q[i+2]
            df2 = row[i+2]
            a = q[47+i]
            b = row[47+i]
            damn = []
        
            if i !=3:
                damn.extend([period[i], merged.avgReturns_merged(a[1], b[1], df1, df2), merged.freq_combined(df1, df2, a[2], b[2]), merged.comb_tradingNum(df1, df2, a[3], b[3]), merged.combined_numLoss(df1, df2, a[4], b[4], 1), merged.combined_numLoss(df1, df2, a[5], b[5], -1)])
                if i ==0:
                    damn.extend([max([a[6], b[6]], key=lambda lst: lst[0]), min([a[7], a[7]], key=lambda lst: lst[0]), merged.merged_max_cons(df1, df2, a[8], b[8], 1), merged.merged_max_cons(df1, df2, a[9], b[9], -1)])
                else:
                    damn.extend([merged.max_min_Profit(1, df1, df2, a[6], b[6], row_combined[i+2]), merged.max_min_Profit(-1, df1, df2, a[7], b[7], row_combined[i+2]) ])
            if i ==3: 
                damn.extend([merged.max_profit(row_combined[5], 1), merged.max_profit(row_combined[5], -1)])
            row_combined.append(damn)    
                        
        for p in pick:
            row_combined[p] = pickle.dumps(row_combined[p])

    #cursor.execute('''INSERT OR REPLACE INTO StrategyData VALUES (?, ?, ?,?,?,?,?,?, ?, ?,?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row_combined)
        cursor.execute('''INSERT OR REPLACE INTO StrategyData VALUES (?, ?, ?,?,?,?,?,?, ?, ?,?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row_combined)
    else:
        print("given id does not exist")   
        cursor.execute('''INSERT OR REPLACE INTO StrategyData VALUES (?, ?, ?,?,?,?,?,?, ?, ?,?,?,?,?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row_pickled)
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

        print(f"Row with ID '{id_to_delete}' has been deleted successfully.")

    except sqlite3.Error as e:
        print("SQLite error:", e)
    
    conn.close()


conn = sqlite3.connect('strategy_analysis.db')
cursor = conn.cursor()

#cursor.execute('''DROP TABLE IF EXISTS StrategyData''')
# # # Create a table
cursor.execute('''CREATE TABLE IF NOT EXISTS StrategyData (
    Id TEXT PRIMARY KEY,
    Csv BLOB,
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
    monthWR REAL,
    weekWR REAL,
    yearWR REAL,
    months6WR REAL,
    quarterWR REAL,
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
    annual_mean REAL,
    disp_Daily BLOB,
    disp_Monthly BLOB,
    disp_Weekly BLOB,
    disp_Weekday BLOB,
    disp_Yearly BLOB
)''')

conn.commit()
conn.close()

# # # # Example usage
head = ["Id","csv","Daily", "Monthly", "Weekly", "Weekday", "Yearly", 'MaxDD',"DDpct", "Aloss", "Awin", "MaxG", "MinG", "short", "long","Atr", "NumWins", "NumLoss", "HIT", "ROI", "PF", "Tot_pos","Tot_neg", "YVola", "MaxWinS", "MaxLossS","daily_annual_std", "daily_annual_mean","Inv","rfr", "EquityPct", "monthWR", "weekWR", "yearWR", "months6WR", "quarterWR", "Sharpe", "Calmar", "Sortino", "threeD", "thirtyD", 'twoW', "sixM","oneY", "twoY", "annual_mean", "annual_std", "disp_daily", "disp_monthly", "disp_weekly", "disp_weekday", "disp_yearly"]
# # daily = ["period", "avgRet", "avgRetPct", "freq", "numD", "numProfit", "numLoss", "MPDay", "MaxG", "LPDay", "MinG", "MaxWinS", "MaxLossS"]

path = "files/StrategyBacktestingPLBook-*.csv"
Files = []

for file in glob.glob(path, recursive=True):
    Files.append(file)
#Files = ["files/StrategyBacktestingPLBook-STAB679.csv"]
for i in Files:
    insert_sql(i)
    
