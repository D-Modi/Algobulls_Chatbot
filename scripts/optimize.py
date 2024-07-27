import streamlit as st
import pandas as pd
import numpy as np
import re
import glob
from stratrgy_analysis import StatergyAnalysis
from dateutil import parser
from customerPLBook_analysis import customerPLBook_Analysis
from calculations import calc
    
def set_page_config():
    if 'page_config_set' not in st.session_state:
        st.set_page_config(layout="wide")
        st.session_state['page_config_set'] = True

def parse_dates(date_str):
    return parser.parse(date_str)

def get_files():
    path = "files/StrategyBacktestingPLBook-*.csv"
    Files = []

    for file in glob.glob(path, recursive=True):
        found = re.search('StrategyBacktestingPLBook(.+?)csv', str(file)).group(1)[1:-1]
        Files.append(found)
    
    return Files

def data_list(options, weights): 
    data = []
    dfs = {}
    head = ['Strategy_Code', 'ROI%', 'HIT Ratio', 'Profit Factor']
    
    for i in options:
        csv_path = f"files/StrategyBacktestingPLBook-{i}.csv"
        row = calc(csv_path)
        data.append([i, row[19][1], row[18], row[20]])
        dfs[i] = row[1]
        
    portfolio = merged_csv(options, dfs, weights)
    Analysis = calc(portfolio, is_dataframe=1, filename="Portfolio")
    data.append([Analysis[0], Analysis[19][1], Analysis[18], Analysis[20]])
    
    df = pd.DataFrame(data, columns=head) 
    df.set_index("Strategy_Code", inplace=True)
    
    st.write(df)
    return portfolio
    
def calc_amt(options, weights, total_investment):
    investment = {}
    for e in options:
        investment[e] = weights[e]*total_investment
        st.session_state['Total_investment'] = total_investment
        st.session_state['investment'] = investment
        st.session_state['weights'] = weights

def calc_weights(options, investment, total_investment):
    weights = {}
    for e in options:
        weights[e] = investment[e]/total_investment
        st.session_state['Total_investment'] = total_investment
        st.session_state['investment'] = investment
        st.session_state['weights'] = weights
    
def get_weights(options):
    if not options:
        st.write("No strategies Seleted")
        return
    
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
            number = st.number_input(f"Enter weights for {strategy}",min_value=0.0, value=equal, key= strategy, placeholder=f"{strategy}_weights")
        else:
            number = st.number_input(f"Enter weights for {strategy}",min_value=0.0, value=st.session_state['weights'][strategy], key= strategy, placeholder=f"{strategy}_RE")
        tot -= number
        sum_weights +=number
        if(j!=len(options)-1 and tot>=0):
            equal = tot/(len(options)-j-1)
        else:
            equal = 0.0
        weights[strategy] = number
        
    if st.button("submit", key= "weights_bt"):
        st.session_state['Entered_values'] = True
        for strategy in options:
            weights[strategy] = weights[strategy]/sum_weights
        calc_amt(options, weights, investment)

def get_investment(options):
    if not options:
        st.write("No strategies Seleted")
        return
    
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
        st.session_state['Entered_values'] = True
        calc_weights(options, amount, total_investment)
    
def merged_csv(options, dfs, weight):
    data = pd.DataFrame()
    
    for i in options:
        df = dfs[i]
        df['pnl_absolute'] = df['pnl_absolute'] * weight[i]
        
        if len(data) == 0:  
            data = df
        else:
            data = pd.concat([df, data], axis=0) 
    
    data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data.sort_values(by='date')
    data.reset_index(inplace=True)
    data['pnl_absolute_cumulative'] = data['pnl_absolute'].cumsum()
    
    return data

def complete_analysis(data, total_investment):
    Analysis = StatergyAnalysis(data, is_dataframe=1, number=total_investment, customerPLBook=True)
    obj =  customerPLBook_Analysis()
    obj.customerPLBook_analysis_display(Analysis, option=None)
    
def run_optimize():       
    set_page_config()
    Files = get_files()
    options = st.multiselect(
        "Selet Trading Stratergies",
        Files)
    
    col1, col2, col3 = st.columns([1.1,1.9,11])
    with col1:
        if st.session_state['show_investments'] == False and  st.session_state['Entered_values'] == False:
            if st.button("Enter Weights"):
                st.session_state['show_weights'] = True
                
    with col2:
        if  st.session_state['show_weights'] == False and st.session_state['show_investments'] ==False and st.session_state['Entered_values'] == False: 
            if st.button("Enter Investment Amounts"):
                st.session_state['show_investments'] = True

    if st.session_state['show_investments'] ==True: 
            if st.button("Enter Investment Amounts", key = "col1"):
                st.session_state['show_investments'] = True
                
    if st.session_state['show_weights']:
        if st.session_state['Entered_values'] == False:
            get_weights(options)
            
        if st.session_state['Entered_values'] == True:
            st.write(st.session_state['weights'])
            st.write("Total investment", st.session_state['Total_investment'])
            
            if(st.button("Re-Enter Weights & initial invetment")):
                st.session_state['Entered_values'] = False

    if st.session_state['show_investments']: 
        if st.session_state['Entered_values'] == False:
            get_investment(options)
            
        if st.session_state['Entered_values'] == True:
            st.write(st.session_state['investment'])
            st.write("Total investment", st.session_state['Total_investment'])
                
            if(st.button("Re-Enter invetment amount")):
                st.session_state['Entered_values'] = False

    if st.session_state['Entered_values'] == True:
        weights = st.session_state['weights']
        investment = st.session_state['investment']
        total_investment = st.session_state['Total_investment']
        
        combined_csv = data_list(options, weights)
        
        if st.session_state['show_Analysis'] == False:
            if(st.button("Show Complete Analysis")):
                st.session_state['show_Analysis'] = True
        if st.session_state['show_Analysis'] == True:
            complete_analysis(combined_csv, total_investment)
                
            
 