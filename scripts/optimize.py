import streamlit as st
import pandas as pd
import numpy as np
import re
import glob
from stratrgy_analysis import StatergyAnalysis
from dateutil import parser
from customerPLBook_analysis import customerPLBook_Analysis

if 'show_investments' not in st.session_state:
    st.session_state['show_investments'] = False 
if 'show_weights' not in st.session_state:
    st.session_state['show_weights'] = False
if 'show_Analysis' not in st.session_state:
    st.session_state['show_Analysis'] = False
if 'Entered_values' not in st.session_state:
    st.session_state['Entered_values'] = False
if 'investment' not in st.session_state:
    st.session_state['investment'] = None
if 'weights' not in st.session_state:
    st.session_state['weights'] = None
if 'Total_investment' not in st.session_state:
    st.session_state['Total_investment'] = None
    
def set_page_config():
    if 'page_config_set' not in st.session_state:
        st.set_page_config(layout="wide")
        st.session_state['page_config_set'] = True
        
set_page_config()

def parse_dates(date_str):
    return parser.parse(date_str)

def get_csv_data(filepath):
    
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
    data['Date'] = data['entry_timestamp'].apply(parse_dates)
    data.drop(columns=['entry_timestamp'], inplace=True)   
    data['Day'] = pd.to_datetime(data.Date,format = '%Y-%m')
    data['Week'] = pd.to_datetime(data.Date,format = '%dd-%m')
    data['Month'] = pd.to_datetime(data.Date,format = '%Y-%m')
    data['Year'] = pd.to_datetime(data.Date,format = '%Y-%m')
    data['weekday'] = pd.to_datetime(data.Date,format = '%a')
    
    data['Day'] = data['Day'].dt.strftime('%Y-%m-%d')
    data['Week'] = data['Week'].dt.strftime('%Y-%U')
    data['Month'] = data['Month'].dt.strftime('%Y-%m')
    data['Year'] = data['Year'].dt.strftime('%Y')
    data['weekday'] = data['weekday'].dt.strftime('%a')
    return data
    
def calc_amt(options, weights, total_investment):
    investment = {}
    for e in options:
        investment[e] = weights[e]*total_investment
        st.session_state['Total_investment'] = total_investment
        st.session_state['investment'] = investment
        st.session_state['weights'] = weights
    
def get_weights(options):
    tot = 100.0
    sum_weights = 0
    weights = {} 
    if st.session_state['Total_investment'] is None:   
        investment = st.number_input(f"Enter Total investent",min_value=10000.0, value=10000.0) 
    else:
        investment = st.number_input(f"Enter Total investent",min_value=10000.0, value=st.session_state['Total_investment'], key="Re_enter_investment")
    equal = 100/len(options)
    
    for j in range(len(options)):
        strategy = options[j]
        if st.session_state['weights'] is None:
            number = st.number_input(f"Enter weights for {strategy}", value=equal, key= strategy, placeholder=f"{strategy}_weights")
        else:
            number = st.number_input(f"Enter weights for {strategy}", value=st.session_state['weights'][strategy], key= strategy, placeholder=f"{strategy}_RE")
        tot -= number
        sum_weights +=number
        if(j!=len(options)-1):
            equal = tot/(len(options)-j-1)
        weights[strategy] = number
        
    if st.button("submit", key= "weights_bt"):
        st.session_state['Entered_values'] = True
        for strategy in options:
            weights[strategy] = weights[strategy]/sum_weights
        calc_amt(options, weights, investment)

def get_investment(options):
    total_investment = 0.0
    amount = {}    
    for i in options:
        if st.session_state['investment'] is None:
            number = st.number_input(f"Enter investent amount for {i}",min_value=0.0, value=0.0, placeholder=f"{i}_amount")
        else:
            number = st.number_input(f"Enter investent amount for {i}",min_value=0.0, value=st.session_state['investment'][i], placeholder=f"{i}_amount_RE")
        total_investment += number
        amount[i] = number
    
    if st.button("submit", key= "investment_bt"):
        st.session_state['Total_investment'] = total_investment
        st.session_state['investment'] = amount
        st.session_state['Entered_values'] = True
    
    
def merged_csv(options, investment):
    data = pd.DataFrame()
    
    for i in options:
        csv_path = f"files/StrategyBacktestingPLBook-{i}.csv"
        df = get_csv_data(csv_path)
        initial_investment =  df['equity_curve'].iat[-1] - df['pnl_cumulative_absolute'].iat[-1]
        df['pnl_absolute'] = df['pnl_absolute']/initial_investment* investment[i]    
        
        if len(data) == 0:  
            data = df
        else:
            data = pd.concat([df, data], axis=0) 
    
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data.sort_values(by='Date')
    data.reset_index(inplace=True)
    data['pnl_absolute_cumulative'] = data['pnl_absolute'].cumsum()
    
    return data

def complete_analysis(data, total_investment):
    Analysis = StatergyAnalysis(data, is_dataframe=1, number=total_investment, customerPLBook=True)
    obj =  customerPLBook_Analysis()
    obj.customerPLBook_analysis_display(Analysis, option=None)
    
def calc_roi(data, total_investment):
    roi = data['pnl_absolute_cumulative'].iat[-1]/total_investment*100
    return roi      
    
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
if st.session_state['show_investments'] == False and  st.session_state['Entered_values'] == False:
    if st.button("Enter Weights"):
        st.session_state['show_weights'] = True

if  st.session_state['show_weights'] == False and  st.session_state['Entered_values'] == False: 
    if st.button("Enter Investment Amounts"):
        st.session_state['show_investments'] = True

if st.session_state['show_weights']:
    try:  
        if st.session_state['Entered_values'] == False:
            get_weights(options)
            
        if st.session_state['Entered_values'] == True:
            st.write(st.session_state['weights'])
            st.write("Total investment", st.session_state['Total_investment'])
            
            investment = st.session_state['investment']
            total_investment = st.session_state['Total_investment']
            combined_csv = merged_csv(options, investment)
            roi = calc_roi(combined_csv, total_investment)
            st.write(f"ROI: {roi}")
            
            if st.session_state['show_Analysis'] == False:
                if(st.button("Show Complete Analysis")):
                    st.session_state['show_Analysis'] = True
            if st.session_state['show_Analysis'] == True:
                complete_analysis(combined_csv, total_investment)
                
            if(st.button("Re-Enter Weights & initial invetment")):
                st.session_state['Entered_values'] = False
    except:    
        print("Waiting")

if st.session_state['show_investments']:
    try:  
        if st.session_state['Entered_values'] == False:
            get_investment(options)
            
        if st.session_state['Entered_values'] == True:
            st.write(st.session_state['investment'])
            st.write("Total investment", st.session_state['Total_investment'])
            
            investment = st.session_state['investment']
            total_investment = st.session_state['Total_investment']
            combined_csv = merged_csv(options, investment)
            roi = calc_roi(combined_csv, total_investment)
            st.write(f"ROI: {roi}")
            
            if st.session_state['show_Analysis'] == False:
                if(st.button("Show Complete Analysis")):
                    st.session_state['show_Analysis'] = True
            if st.session_state['show_Analysis'] == True:
                complete_analysis(combined_csv, total_investment)
                
            if(st.button("Re-Enter invetment amount")):
                st.session_state['Entered_values'] = False
    except:    
        print("Waiting")


