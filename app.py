# src/app/app.py

import streamlit as st

def home_page():
    st.title("Machine Learning Model Training App")
    st.write(
        "Welcome to the Machine Learning Model Training App! This is a simple Streamlit app "
        "that allows you to upload data, choose an ML model, and view the training results."
    )

    # Add more content or instructions as needed

def data_upload_page():
    st.title("Data Upload Page")
    # Add your data upload logic here

def model_selection_page():
    st.title("Model Selection Page")
    # Add your model selection logic here

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
