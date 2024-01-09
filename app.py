# src/app/app.py

import streamlit as st
import os
import sys
import base64
import time
import dill
import pandas as pd
import matplotlib.pyplot as plt

from src.components.data_ingestion import DataIngestionConfig,DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.pipeline.train_pipeline import TrainPipeline
from src.exception import CustomException
from src.utils import load_object


def home_page():
    st.title("Thermal Error Prediction in Machining")
    st.write(
        "Welcome to the Machine Learning Model Training UI! This is a simple UI "
        "that allows you to upload data, choose an ML model, and view the training results."
    )

def data_upload_page():
    st.title("Data Upload")

    st.write(
        "This page allows you to upload the temperature and thermal error data in '.xlsx' format. "
        "You can also preview the data and see the visualizations of your preferred day's data. "
    )

    # Upload temperature data file
    uploaded_temp_file = st.file_uploader("Upload Temperature data", type=["xlsx"])
    temp_file_path = handle_uploaded_file(uploaded_temp_file, "temperature")

    # Upload thermal displacement data file
    uploaded_disp_file = st.file_uploader("Upload Thermal Displacement data", type=["xlsx"])
    disp_file_path = handle_uploaded_file(uploaded_disp_file, "thermal_displacement")

    if temp_file_path and disp_file_path:

        temp_xls = pd.ExcelFile(temp_file_path)
        num_days = len(temp_xls.sheet_names)

        day_options = [f"Day {i + 1}" for i in range(num_days)]
        selected_day = st.selectbox("Select Day to preview", day_options)
        temp_df = temp_xls.parse(temp_xls.sheet_names[day_options.index(selected_day)])

        disp_xls = pd.ExcelFile(disp_file_path)
        disp_df = disp_xls.parse(disp_xls.sheet_names[day_options.index(selected_day)])

        st.session_state.temp_df = temp_df
        st.session_state.disp_df = disp_df
        st.session_state.day = selected_day

    if 'temp_df' in st.session_state and 'disp_df' in st.session_state and 'day' in st.session_state:

        selected_day = st.session_state.day
        st.subheader(f"{selected_day}: Temperature Data Preview")
        st.dataframe(st.session_state.temp_df)

        st.subheader(f"{selected_day}: Thermal Displacement Data Preview")
        st.dataframe(st.session_state.disp_df)

        st.subheader(f"{selected_day}: Temperature Data Plot")
        plot_data(st.session_state.temp_df, "Temperature")

        st.subheader(f"{selected_day}: Thermal Displacement Data Plot")
        plot_data(st.session_state.disp_df, "Thermal Displacement")

def handle_uploaded_file(uploaded_file, file_type):
    if uploaded_file:
        save_directory = "uploaded_data"
        os.makedirs(save_directory, exist_ok=True)

        file_path = os.path.join(save_directory, f"{file_type}_data.{uploaded_file.name.split('.')[-1]}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        return file_path

def plot_data(df, title):

    plt.figure(figsize=(12, 8))

    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.title(f"{title} Data")
    plt.xlabel("Samples")
    if title == 'Temperature':
        plt.ylabel(f"{title} in °C")
    else:
        plt.ylabel(f"{title} in microns")
    plt.legend()
    st.pyplot(plt)

def model_selection_page():
    st.title("ML Model Training")

    st.write(
        "This page allows you to choose the ML model you want to train and train-validation split. "
        "The chosen model is then fit on the training set after hyperparameter tuning and the model "
        "is then used to make predictions on the validation set. "
        "The Results of training and validation are then displayed. You can also download the pickle file of the model."

    )

    models = [
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting",
        "Linear Regression",
        "XGBRegressor",
        "CatBoosting Regressor",
        "AdaBoost Regressor",
    ]

    selected_model = st.selectbox("Select a Model", models)
    st.write(f"You selected: {selected_model}")

    validation_split = 0.01*int(st.slider("Select Validation Split (%)", 0, 100, 20, step=10))

    with st.spinner(f"Training the model..."):
        time.sleep(5)  

        model_name = str(selected_model)

        train_pipeline = TrainPipeline()
        report = train_pipeline.train_selected_model(model_name,validation_split)

        st.success("Model training completed!")

        save_model_path = os.path.join(os.path.join('artifacts','trained_models'),f"{model_name}_model.pkl")

        st.subheader("Download Trained Model:")
        st.markdown(get_download_link(save_model_path, f"{model_name}.pkl"), unsafe_allow_html=True)

        st.subheader(f"{model_name}: Training Results:")
        st.write(f"Training RMSE: {report[model_name]['train_rmse']:.4f} microns")
        st.write(f"Training R2: {report[model_name]['train_r2']:.4f}")

        model_training_results(selected_model, report)
        model_validation_results(selected_model, report)


def model_comparison_page():
    st.title('Model Comparison')

    st.write(
        "This page trains all the model and shows a comparison between all the models. "
        "This comparison of performance of different models are then used to choose the "
        "best model for deployment."
    )
    
    st.subheader('Models Being Trained:')

    models = [
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting",
        "Linear Regression",
        "XGBRegressor",
        "CatBoosting Regressor",
        "AdaBoost Regressor",
    ]
    for model in models:
        st.write(f"- {model}")

    validation_split = 0.01*int(st.slider("Select Validation Split (%)", 0, 100, 20, step=10))

    with st.spinner(f"Training the model..."):
        time.sleep(5)  
        
        train_pipeline = TrainPipeline()
        report = train_pipeline.train_all_models(validation_split)

        st.success("Model training completed!")

        st.subheader('Results:')
        st.dataframe(report)

        model_name = report['Test RMSE'].idxmin()

        st.write(f'Based on the comparison of various models mentioned above, {model_name} has been chosen as the best model for deployment. You can download the pickle file of the model from the link provided below.')
        save_model_path = os.path.join(os.path.join('artifacts','trained_models'),f"{model_name}_model.pkl")

        st.subheader("Download Best Model:")
        st.markdown(get_download_link(save_model_path, f"{model_name}.pkl"), unsafe_allow_html=True)


def deep_learning_page():
    st.title('Deep Learning models')

    deep_learning_models = ["DNN (Deep Neural Networks)"]
    selected_model = st.selectbox("Select a Model", deep_learning_models)
    st.write(f"You selected: {selected_model}")

    validation_split = 0.01*int(st.slider("Select Validation Split (%)", 0, 100, 20, step=10))

    with st.spinner(f"Training the model..."):
        time.sleep(5)  

        obj = DataIngestion()
        X_train_path,Y_train_path,X_test_path,Y_test_path,_,_ = obj.initiate_data_ingestion(validation_split)

        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(X_train_path,Y_train_path,X_test_path,Y_test_path)

        model_trainer = ModelTrainer()
        model_name = str(selected_model)
        report = model_trainer.initiate_dl_training(model_name,train_arr,test_arr)

        st.success("Model training completed!")

        save_model_path = os.path.join(os.path.join('artifacts','trained_models'),f"{model_name}_model.pkl")

        st.subheader("Download Trained Model:")
        st.markdown(get_download_link(save_model_path, f"{model_name}.pkl"), unsafe_allow_html=True)

        st.subheader(f"{model_name}: Training Results:")
        st.write(f"Training RMSE: {report[model_name]['train_rmse']:.4f} microns")
        st.write(f"Training R2: {report[model_name]['train_r2']:.4f}")

        # Parity plot for Training
        fig, ax = plt.subplots()
        ax.scatter(report[model_name]['y_train'], report[model_name]['y_train_pred'])
        ax.plot([min(report[model_name]['y_train']), max(report[model_name]['y_train'])],
                [min(report[model_name]['y_train']), max(report[model_name]['y_train'])], 'k--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Parity Plot - Training Set')
        st.pyplot(fig)

        st.subheader(f"{model_name}: Validation Results:")
        st.write(f"Validation RMSE: {report[model_name]['test_rmse']:.4f} microns")
        st.write(f"Validation R2: {report[model_name]['test_r2']:.4f}")

        # Parity plot for Validation
        fig, ax = plt.subplots()
        ax.scatter(report[model_name]['y_test'], report[model_name]['y_test_pred'])
        ax.plot([min(report[model_name]['y_test']), max(report[model_name]['y_test'])],
                [min(report[model_name]['y_test']), max(report[model_name]['y_test'])], 'k--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Parity Plot - Validation Set')
        st.pyplot(fig)


def continual_learning_page():
    st.title('Continual Learning models')

    continual_learning_models = ["DNN (Deep Neural Networks) - Daywise Training"]
    selected_model = st.selectbox("Select a Model", continual_learning_models)
    st.write(f"You selected: {selected_model}")

    data_ingestion = DataIngestion()
    num_days = data_ingestion.initiate_day_wise_data_ingestion()
    validation_split = int(st.slider("Select Days for validation", 0, num_days, int(0.2*num_days), step=1))

    st.write(f"{num_days-validation_split} days of data will be used for training the model and {validation_split} days of data will be used for validation.")


def make_predictions_page():
    st.title("Get Predictions")

    uploaded_model = st.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])
    input_file = st.file_uploader("Upload temperature data for prediction",type=['csv'])

    if uploaded_model is not None and input_file is not None:
        try:
            model = dill.load(uploaded_model)
            st.success("Model loaded successfully!")

        except Exception as e:
            st.error("Error loading the model!")
            raise CustomException(e,sys)
        
        try:
            input_data = pd.read_csv(input_file)
            st.success("Data loaded successfully!")

        except Exception as e:
            st.error("Error loading the data!")
            raise CustomException(e,sys)
        
        preprocessor_path = os.path.join('artifacts','preprocesser.pkl')
        preproceesor = load_object(file_path = preprocessor_path) 
        scaled_input = preproceesor.transform(input_data)

        predictions = pd.DataFrame(model.predict(scaled_input),columns=['Error (in microns)'])

        st.subheader('Your Input data:')
        st.write(input_data)

        st.subheader('Predictions:')
        st.write(predictions)



def plot_parity(ax, actual, predicted, title):
    ax.scatter(actual, predicted)
    ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Parity Plot - {title} Set')

def model_training_results(selected_model, report):
    st.subheader(f"{selected_model}: Training Results:")
    st.write(f"Training RMSE: {report[selected_model]['train_rmse']:.4f} microns")
    st.write(f"Training R2: {report[selected_model]['train_r2']:.4f}")

    fig, ax = plt.subplots()
    plot_parity(ax, report[selected_model]['y_train'], report[selected_model]['y_train_pred'], 'Training')
    st.pyplot(fig)

def model_validation_results(selected_model, report):
    st.subheader(f"{selected_model}: Validation Results:")
    st.write(f"Validation RMSE: {report[selected_model]['test_rmse']:.4f} microns")
    st.write(f"Validation R2: {report[selected_model]['test_r2']:.4f}")

    fig, ax = plt.subplots()
    plot_parity(ax, report[selected_model]['y_test'], report[selected_model]['y_test_pred'], 'Validation')
    st.pyplot(fig)

def get_download_link(file_path, text):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f"<a href='data:application/octet-stream;base64,{b64}' download='{os.path.basename(file_path)}'>{text}</a>"
    return href


def main():
    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Data Upload": data_upload_page,
        "ML Model Training": model_selection_page,
        "ML Model Comparison": model_comparison_page,
        "Get Predictions": make_predictions_page,
        "Deep Learning models": deep_learning_page,
        "Continual Learning models": continual_learning_page,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

if __name__ == "__main__":
    main()
