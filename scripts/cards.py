import streamlit as st
from streamlit.errors import StreamlitAPIException
from customerPLBook_analysis import customerPLBook_Analysis
from utils import *
from optimize import run_optimize, reset    
from login_pg import login
from admin_pg import admin_view

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
if 'columns' not in st.session_state:
    st.session_state['columns'] = [1,1,2.5,3,6,1.5]
if 'account_details' not in st.session_state:
    st.session_state['account_details'] = None
if 'first_date' not in st.session_state:
    st.session_state['first_date'] = None
if 'last_date' not in st.session_state:
    st.session_state['last_date'] = None
if 'complete_df' not in st.session_state:
    st.session_state['complete_df'] = None
if 'initial_investment' not in st.session_state:
    st.session_state['initial_investment'] = 150000
    
def CustomerPLBook(customerPLBook_analysis_streamlit):
    flag = 0
    flag = customerPLBook_analysis_streamlit.run()
    if st.session_state['sidebar'] == "CustomerPLBook" and flag == 1:
        customerPLBook_analysis_streamlit.sidebar()
        
def click_button_signout():
        st.session_state['sidebar']= "Home"
        st.session_state['account_details'] = None
        st.session_state['columns'] = [1,1,2.5,3,6,1.5]
        
set_page_config()
customerPLBook_analysis_streamlit = customerPLBook_Analysis()
if st.session_state['account_details'] is None:
    details = login()
    if details is not None:
        if details['admin'] == "No":
            st.session_state['columns'][4] += st.session_state['columns'][0] - 0.1
            st.session_state['columns'][0] = 0.1
        if details['view_cards'] == "No":
            st.session_state['columns'][4] += st.session_state['columns'][1] - 0.1
            st.session_state['columns'][1] = 0.1
            if details['user_PL_book'] == "Yes":
                st.session_state['sidebar']= "CustomerPLBook"
            elif details['use_combination'] == "Yes":
                st.session_state['sidebar']= "PortfolioOptimization"
                reset()
            else :
                st.session_state['sidebar']=None
        if details['use_combination'] == "No":
            st.session_state['columns'][4] += st.session_state['columns'][3] - 0.1
            st.session_state['columns'][3] = 0.1                 
        if details['user_PL_book'] == "No":
            st.session_state['columns'][4] += st.session_state['columns'][2] - 0.1
            st.session_state['columns'][2] = 0.1
        if details['admin'] == "Yes":
            st.session_state['sidebar']= "Admin"
            
        st.session_state['account_details'] = details
else: 
    st.session_state["run_once"] = True
    col0, col1, col2 ,col3, col4, col5= st.columns(st.session_state['columns'])
    
    with col0:
        if st.session_state['columns'][0] != 0.1:  
            if st.button("Admin", use_container_width=True):   
                st.session_state['sidebar']="Admin"
                
    with col1:           
        if st.session_state['columns'][1] != 0.1:
            if st.button("Home", use_container_width=True):
                if(st.session_state['sidebar'] == "Home"):
                    if(st.session_state.clicked):
                        click_button_return()
                else:
                    st.session_state['sidebar']="Home"
            
    with col2:              
        if st.session_state['columns'][2] != 0.1:       
            if st.button("CustomerPLBook Analysis", use_container_width=True):
                st.session_state['sidebar']="CustomerPLBook"
    
                
    with col3: 
        if st.session_state['columns'][3] != 0.1:   
            if(st.button("Portfolio Optimiation")):
                reset()
                st.session_state['sidebar']="PortfolioOptimization" 
    with col5:
        st.button("Sign Out", on_click=click_button_signout, use_container_width=True)
                    
    if st.session_state['sidebar'] is None:
        st.write("You Don't Have the required Permission.For Further details, contact Admin.")
    if st.session_state['sidebar']=="Home":
        home()
    if st.session_state['sidebar']=="CustomerPLBook":
        CustomerPLBook(customerPLBook_analysis_streamlit)
    if st.session_state['sidebar']=="PortfolioOptimization":
        run_optimize()
    if st.session_state['sidebar']=="Admin":
        admin_view()