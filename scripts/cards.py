import streamlit as st
from streamlit.errors import StreamlitAPIException
from customerPLBook_analysis import customerPLBook_Analysis
from utils import *
from optimize import run_optimize, reset       

if 'sidebar' not in st.session_state:
    st.session_state['sidebar']= "Home" 
if 'clicked' not in st.session_state:
    st.session_state.clicked = False 
if 'ana' not in st.session_state:
    st.session_state['ana'] = None
if 'stra' not in st.session_state:
    st.session_state['stra'] = None
if 'button' not in st.session_state:
    st.session_state.button = False 
if 'Time' not in st.session_state:
    st.session_state['Time'] = None
if 'rt' not in st.session_state:
    st.session_state['rt'] = None
if 'i' not in st.session_state:
    st.session_state['i'] = None   
if 'index' not in st.session_state:
    st.session_state['index'] = None 
if 'warning_message' not in st.session_state:
    st.session_state.warning_message = ""
if 'new_q' not in st.session_state:
    st.session_state['new_q'] = None
if 'show_investments' not in st.session_state:
    st.session_state['show_investments'] = False 
if 'show_weights' not in st.session_state:
    st.session_state['show_weights'] = False
if 'show_Analysis' not in st.session_state:
    st.session_state['show_Analysis'] = False
if 'Entered_values' not in st.session_state:
    st.session_state['Entered_values'] = None
if 'investment' not in st.session_state:
    st.session_state['investment'] = None
if 'weights' not in st.session_state:
    st.session_state['weights'] = None
if 'Total_investment' not in st.session_state:
    st.session_state['Total_investment'] = None
if 'options' not in st.session_state:
    st.session_state['options'] = None

set_page_config()

def CustomerPLBook(customerPLBook_analysis_streamlit):
    flag = 0
    flag = customerPLBook_analysis_streamlit.run()
    if st.session_state['sidebar'] == "CustomerPLBook" and flag == 1:
        customerPLBook_analysis_streamlit.sidebar()


customerPLBook_analysis_streamlit = customerPLBook_Analysis()
col1, col2 , col3,col4= st.columns([1,2.5,3,8.5])
with col1:
    if st.button("Home", use_container_width=True):
        if(st.session_state['sidebar'] == "Home"):
            if(st.session_state.clicked):
                click_button_return()
        else:
            st.session_state['sidebar']="Home"
        
with col2:
    if st.button("CustomerPLBook Analysis", use_container_width=True):
        st.session_state['sidebar']="CustomerPLBook"

with col3:  
    if(st.button("Portfolio Optimiation")):
        reset()
        st.session_state['sidebar']="PortfolioOptimization"
        
if st.session_state['sidebar']=="Home":
    home()
if st.session_state['sidebar']=="CustomerPLBook":
    CustomerPLBook(customerPLBook_analysis_streamlit)
if st.session_state['sidebar']=="PortfolioOptimization":
    run_optimize()
    