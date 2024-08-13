import streamlit as st
import pandas as pd
import os

# Path to the CSV file
CSV_PATH = os.path.join(os.getcwd(), 'user_data', 'users.csv')

# Function to load user data from CSV
def load_user_data():
    return pd.read_csv(CSV_PATH)

# Function to save user data to CSV
def save_user_data(df):
    df.to_csv(CSV_PATH, index=False)

# Function to delete a user
def delete_user(user_data, index):
    user_data = user_data.drop(index).reset_index(drop=True)
    save_user_data(user_data)

# Streamlit app
def main():
    st.title("Admin Page")

    # Load user data
    user_data = load_user_data()

    # Display the user data in a table format
    st.subheader("User Data")
    st.write(user_data)

    # Radio buttons to select the operation, placed inline
    st.subheader("Select Action")
    action = st.radio("", options=["Add User", "Edit User", "Delete User"], horizontal=True)
    dropdown_options = ["Yes", "No"]
    if action == "Add User":
        st.subheader("Add New User")
        with st.form(key='add_user_form'):
            email_id = st.text_input("Email_ID")
            password = st.text_input("Password", type="password")
            add_cards = st.selectbox("Add Cards", dropdown_options)
            view_cards = st.selectbox("View Cards", dropdown_options)
            use_combination = st.selectbox("Use Combination", dropdown_options)
            user_PL_book = st.selectbox("User PL Book", dropdown_options)

            submit_button = st.form_submit_button("Add User")
            if submit_button:
                if email_id and password:
                    user_data = load_user_data()  # Reload to include any new data
                    if email_id in user_data['Email_ID'].values:
                        st.error("User with this email ID already exists!")
                    else:
                        new_user = {
                            'Email_ID': email_id,
                            'Password': password,
                            'add_cards': add_cards,
                            'view_cards': view_cards,
                            'use_combination': use_combination,
                            'user_PL_book': user_PL_book
                        }
                        user_data = user_data.append(new_user, ignore_index=True)
                        save_user_data(user_data)
                        st.success("User added successfully!")
                        st.experimental_rerun()

    elif action == "Edit User":
        st.subheader("Edit User")

        user_to_edit = st.selectbox("Select user to edit", user_data['Email_ID'].tolist())
        if user_to_edit:
            user_details = user_data[user_data['Email_ID'] == user_to_edit].iloc[0]

            with st.form(key='edit_form'):
                email_id = st.text_input("Email_ID", value=user_details['Email_ID'], key='edit_email')
                password = st.text_input("Password", value=user_details['Password'], type='password', key='edit_password')
                index = dropdown_options.index(user_details['add_cards'])
                add_cards = st.selectbox("Add Cards", dropdown_options, index=index)
                index = dropdown_options.index(user_details['view_cards'])
                view_cards = st.selectbox("View Cards", dropdown_options, index=index)
                index = dropdown_options.index(user_details['use_combination'])
                use_combination = st.selectbox("Use Combination", dropdown_options, index=index)
                index = dropdown_options.index(user_details['user_PL_book'])
                user_PL_book = st.selectbox("User PL Book", dropdown_options, index=index)
                
                submit_button = st.form_submit_button("Update User")
                if submit_button:
                    index = user_data[user_data['Email_ID'] == email_id].index[0]
                    user_data.at[index, 'Password'] = password
                    user_data.at[index, 'add_cards'] = add_cards
                    user_data.at[index, 'view_cards'] = view_cards
                    user_data.at[index, 'use_combination'] = use_combination
                    user_data.at[index, 'user_PL_book'] = user_PL_book
                    save_user_data(user_data)
                    st.success("User updated successfully!")
                    st.experimental_rerun()

    elif action == "Delete User":
        st.subheader("Delete User")

        user_to_delete = st.selectbox("Select user to delete", user_data['Email_ID'].tolist())
        if user_to_delete:
            if st.button("Delete User"):
                index = user_data[user_data['Email_ID'] == user_to_delete].index[0]
                delete_user(user_data, index)
                st.success("User deleted successfully!")
                st.experimental_rerun()

if __name__ == "__main__":
    main()
