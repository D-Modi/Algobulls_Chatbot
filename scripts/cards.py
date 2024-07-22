import streamlit as st
from customerPLBook_analysis import customerPLBook_Analysis
from utils import *

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

def CustomerPLBook(customerPLBook_analysis_streamlit):
    flag = 0
    flag = customerPLBook_analysis_streamlit.run()
    if st.session_state['sidebar'] == "CustomerPLBook" and flag == 1:
        customerPLBook_analysis_streamlit.sidebar()


customerPLBook_analysis_streamlit = customerPLBook_Analysis()
col1, col2 , col3= st.columns([1,3,10])
with col1:
    if st.button("Home", use_container_width=True):
        if(st.session_state['sidebar'] == "Home"):
            click_button_return()
        else:
            st.session_state['sidebar']="Home"
        

with col2:
    if st.button("CustomerPLBook Analysis"):
        st.session_state['sidebar']="CustomerPLBook"
        
if st.session_state['sidebar']=="Home":
    home()
if st.session_state['sidebar']=="CustomerPLBook":
    CustomerPLBook(customerPLBook_analysis_streamlit)
    
