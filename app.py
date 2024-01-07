# src/app/app.py

import streamlit as st
import os
import pandas as pd

def home_page():
    st.title("Machine Learning Model Training App")
    st.write(
        "Welcome to the Machine Learning Model Training App! This is a simple Streamlit app "
        "that allows you to upload data, choose an ML model, and view the training results."
    )

    # Add more content or instructions as needed

def data_upload_page():
    st.title("Data Upload Page")

    uploaded_file1 = st.file_uploader("Upload temperature data", type=["csv", "xlsx"])
    uploaded_file2 = st.file_uploader("Upload thermal displacement data", type=["csv", "xlsx"])

    if uploaded_file1 and uploaded_file2:

        # df1 = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith('.csv') else pd.read_excel(uploaded_file1, engine='openpyxl')
        # st.write(f"**{uploaded_file1.name}** - {df1.shape[0]} rows x {df1.shape[1]} columns")

        # df2 = pd.read_csv(uploaded_file2) if uploaded_file2.name.endswith('.csv') else pd.read_excel(uploaded_file2, engine='openpyxl')
        # st.write(f"**{uploaded_file2.name}** - {df2.shape[0]} rows x {df2.shape[1]} columns")

        save_directory = "uploaded_data"
        os.makedirs(save_directory, exist_ok=True)

        file_path1 = os.path.join(save_directory, f"uploaded_data_1.{uploaded_file1.name.split('.')[-1]}")
        with open(file_path1, "wb") as f1:
            f1.write(uploaded_file1.getvalue())

        file_path2 = os.path.join(save_directory, f"uploaded_data_2.{uploaded_file2.name.split('.')[-1]}")
        with open(file_path2, "wb") as f2:
            f2.write(uploaded_file2.getvalue())

        st.success(f"Files saved to {save_directory} directory.")

def model_selection_page():
    st.title("Model Selection Page")

def main():
    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Data Upload": data_upload_page,
        "Model Selection": model_selection_page,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page function
    pages[selection]()

if __name__ == "__main__":
    main()
