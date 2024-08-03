import streamlit as st
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import re
import glob

if 'show_investments' not in st.session_state:
    st.session_state['show_investments'] = False 
if 'show_weights' not in st.session_state:
    st.session_state['show_weights'] = False  
 
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
        weights[strategy] = number
        
    if st.button("submit", key= "weights"):
        return weights, investment

def get_investment(options):
    total_investment = 0.0
    d = {}    
    for i in options:
        number = st.number_input(f"Insert investent amount for {i}",min_value=0.0, value=0.0, placeholder=f"{i}_amount")
        total_investment += number
        d[i] = number
    
    if st.button("submit", key= "investment"):
        return d, total_investment

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
        weights, total_investment= get_weights(options)
        st.write("You selected weights:", weights)
        st.write("You selected at:", total_investment)
    except:    
        print("Waiting")

if st.session_state['show_investments']:
    try:
        investments, total_investment = get_investment(options)
        st.write("You selected investment amounts:", investments)
    except:    
        print("Waiting")



