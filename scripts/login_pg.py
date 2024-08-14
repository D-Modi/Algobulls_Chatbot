import streamlit as st
import pandas as pd 
import os

# Function to load user data from CSV
def load_user_data(csv_path):
    return pd.read_csv(csv_path)

# Function to check login credentials
def check_login(email_id, password, user_data):
    user_row = user_data.loc[user_data['Email_ID'] == email_id]
    if not user_row.empty and user_row['Password'].values[0] == password:
        details = {}
        details['view_cards'] = user_row['view_cards'].values[0]
        details['use_combination'] = user_row['use_combination'].values[0]
        details['add_cards'] = user_row['add_cards'].values[0]
        details['user_PL_book'] = user_row['user_PL_book'].values[0]
        details['admin'] = user_row['admin'].values[0]
        return True, details
    return False, None

# Streamlit app
def login():
    st.title("")

    # Center the container by using empty columns on either side
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("Login Page")
        
        # Load user data
        csv_path = os.path.join(os.getcwd(), 'user_data', 'users.csv')
        user_data = load_user_data(csv_path)
        
        # Input fields inside the centered column
        username = st.text_input("Email_ID")
        password = st.text_input("Password", type="password")
        
        # Login button inside the centered column
        if st.button("Login"):
            check, details =  check_login(username, password, user_data)
            if check:
                return details
            else:
                st.error("Invalid username or password.")
                return None
