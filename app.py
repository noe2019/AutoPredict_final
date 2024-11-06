import streamlit as st
import pandas as pd
import os
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
from ydata_profiling import ProfileReport  # Replacement for pandas_profiling

# Check if dataset exists locally
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar configuration
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML Task Generalizer")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This app helps you build and explore models for classification and regression tasks.")

# Upload data
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.columns = df.columns.str.strip()  # Strip whitespace from column names
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

# Profiling data
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if 'df' in locals():
        profile = ProfileReport(df, minimal=True)
        profile.to_file("profile_report.html")
        st.write("The profiling report is ready. Download it to view:")
        with open("profile_report.html", "rb") as f:
            st.download_button("Download Profiling Report", f, file_name="profile_report.html")
    else:
        st.warning("Please upload a dataset first.")

# Model setup and comparison for classification or regression
if choice == "Modelling":
    if 'df' in locals():
        st.title("Model Training")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)

        if chosen_target in df.columns:
            # Remove whitespace from column names
            df.columns = df.columns.str.strip()
            chosen_target = chosen_target.strip()

            # Handle missing values in target column
            if df[chosen_target].isnull().sum() > 0:
                st.warning(f"{df[chosen_target].isnull().sum()} missing values in '{chosen_target}' will be removed.")
                df = df.dropna(subset=[chosen_target])

            # Task type determination based on target variable data type
            if df[chosen_target].dtype == 'object' or df[chosen_target].nunique() <= 20:
                task = 'classification'
            else:
                task = 'regression'

            st.info(f"Detected task type: **{task.capitalize()}** based on target column '{chosen_target}'.")

            if st.button(f'Run {task.capitalize()} Modelling'):
                # Setup and model comparison
                try:
                    if task == 'classification':
                        cls_setup(
                            data=df,  # Pass the full dataframe including target
                            target=chosen_target,
                            session_id=123,  # For reproducibility
                            log_experiment=False,
                            verbose=False
                        )
                        setup_df = cls_pull()
                        st.dataframe(setup_df)

                        # Model comparison and best model selection
                        best_model = cls_compare()
                        compare_df = cls_pull()
                        st.dataframe(compare_df)

                        # Save the best model
                        cls_save(best_model, 'best_model')
                    else:  # Regression setup
                        reg_setup(
                            data=df,  # Pass the full dataframe including target
                            target=chosen_target,
                            session_id=123,  # For reproducibility
                            log_experiment=False,
                            verbose=False
                        )
                        setup_df = reg_pull()
                        st.dataframe(setup_df)

                        # Model comparison and best model selection
                        best_model = reg_compare()
                        compare_df = reg_pull()
                        st.dataframe(compare_df)

                        # Save the best model
                        reg_save(best_model, 'best_model')

                    st.success("Model training complete. You can download the trained model.")
                except ValueError as e:
                    st.error(f"Error in model setup: {str(e)}")
        else:
            st.warning("The chosen target column is not in the dataset.")
    else:
        st.warning("Please upload a dataset first.")

# Model download option
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model available. Please run modelling first.")