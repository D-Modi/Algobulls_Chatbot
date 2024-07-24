import streamlit as st
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import re
import glob
from stratrgy_analysis import StatergyAnalysis

if 'show_investments' not in st.session_state:
    st.session_state['show_investments'] = False 
if 'show_weights' not in st.session_state:
    st.session_state['show_weights'] = False  
    
def get_csv(filepath):
    
    data = pd.read_csv(filepath)
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
    data['date'] = pd.to_datetime(data['entry_timestamp'])
    start = data['date'].iloc[0]
    end = data['date'].iloc[-1]
    if start > end:
        data = data.iloc[::-1].reset_index(drop=True)    
    data = data.reset_index()
    return data
    
def calc_amt(options, weights, total_investment):
    invetment = {}
    for e in options:
        invetment[e] = weights[e]*total_investment
    return invetment, total_investment

def get_weights(options):
    tot = 100.0
    weights = {}   
    investment = st.number_input(f"Insert Total investent",min_value=10000.0, value=10000.0) 
    equal = 100/len(options)
 
    for j in range(len(options)):
        strategy = options[j]
        number = st.number_input(f"Insert weights for {strategy}",min_value=0.01, max_value=tot , value=equal, key= strategy, placeholder=f"{strategy}_weights")
        tot -= number
        if(j!=len(options)-1):
            equal = tot/(len(options)-j-1)
        weights[strategy] = number/100
        
    if st.button("submit", key= "weights"):
        return calc_amt(options, weights, investment)

def get_investment(options):
    total_investment = 0.0
    amount = {}    
    for i in options:
        number = st.number_input(f"Insert investent amount for {i}",min_value=0.0, value=0.0, placeholder=f"{i}_amount")
        total_investment += number
        amount[i] = number
    
    if st.button("submit", key= "investment"):
        return amount, total_investment
    
def merged_csv(options, invetment):
    data = pd.DataFrame()
    for i in options:
        csv_path = f"files/StrategyBacktestingPLBook-{i}.csv"
        df = get_csv(csv_path)
        initial_investment =  df['equity_curve'].iat[-1] - df['pnl_cumulative_absolute'].iat[-1]
        df['pnl_absolute'] = df['pnl_absolute']/initial_investment*invetment[i]
        if len(data) == 0:  
            data = df
        else:
            pd.concat([df, data], axis=0)    
    data.sort_values(by='entry_timestamp')
    return data

def roi_calc(options, investment, total_investment):
    total_profit = 0
    for i in options:
        csv_path = f"files/StrategyBacktestingPLBook-{i}.csv"
        df = get_csv(csv_path)
        initial_investment =  df['equity_curve'].iat[-1] - df['pnl_cumulative_absolute'].iat[-1]
        returns = df['pnl_cumulative_absolute'].iat[-1]/initial_investment*investment[i]
        total_profit +=returns
    return total_profit/total_investment*100        
    
def get_files():
    path = "files/StrategyBacktestingPLBook-*.csv"
    Files = []

    for file in glob.glob(path, recursive=True):
        found = re.search('StrategyBacktestingPLBook(.+?)csv', str(file)).group(1)[1:-1]
        Files.append(found)
    
    return Files

Files = get_files()

options = st.multiselect(
    "Selet Trading Stratergies",
    Files)

st.write("You selected:", options)
if st.session_state['show_investments'] == False:
    if st.button("Enter Weights"):
        st.session_state['show_weights'] = True

if  st.session_state['show_weights'] == False: 
    if st.button("Enter Investment Amounts"):
        st.session_state['show_investments'] = True

if st.session_state['show_weights']:
    try:
        invetment, total_investment= get_weights(options)
        combined_roi = roi_calc(options, invetment, total_investment)
        st.write(f"ROI: {combined_roi}")
        
    except:    
        print("Waiting")

if st.session_state['show_investments']:
    try:
        invetment, total_investment = get_investment(options)
        combined_roi = roi_calc(options, invetment, total_investment)
        st.write(f"ROI: {combined_roi}")
    except:    
        print("Waiting")



