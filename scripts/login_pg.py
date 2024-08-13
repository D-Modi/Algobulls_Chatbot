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
        return True
    return False

# Streamlit app
def main():
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
            if check_login(username, password, user_data):
                st.success(f"Welcome, {username}!")
                st.write("You have successfully logged in.")
            else:
                st.error("Invalid username or password.")

if __name__ == "__main__":
    main()
