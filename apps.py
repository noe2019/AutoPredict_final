import streamlit as st
import pandas as pd
import os
import tempfile
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save
from ydata_profiling import ProfileReport

# Initialize the dataset in session state
if 'df' not in st.session_state:
    st.session_state['df'] = None

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
        df.columns = df.columns.str.strip()
        st.session_state['df'] = df
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)

# Profiling data
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    df = st.session_state['df']
    if df is not None:
        profile = ProfileReport(df, minimal=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            profile.to_file(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                st.download_button("Download Profiling Report", f, file_name="profile_report.html")
    else:
        st.warning("Please upload a dataset first.")

# Model setup and comparison for classification or regression
if choice == "Modelling":
    df = st.session_state['df']
    if df is not None:
        st.title("Model Training")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)

        if chosen_target:
            df.columns = df.columns.str.strip()
            chosen_target = chosen_target.strip()

            if df[chosen_target].isnull().sum() > 0:
                st.warning(f"{df[chosen_target].isnull().sum()} missing values in '{chosen_target}' will be removed.")
                df = df.dropna(subset=[chosen_target])

            task = 'classification' if df[chosen_target].dtype == 'object' or df[chosen_target].nunique() <= 20 else 'regression'
            st.info(f"Detected task type: **{task.capitalize()}** based on target column '{chosen_target}'.")

            if st.button(f'Run {task.capitalize()} Modelling'):
                try:
                    if task == 'classification':
                        cls_setup(data=df, target=chosen_target, session_id=123, log_experiment=False, verbose=False)
                        st.dataframe(cls_pull())
                        best_model = cls_compare()
                        st.dataframe(cls_pull())
                        cls_save(best_model, 'best_model')
                    else:
                        reg_setup(data=df, target=chosen_target, session_id=123, log_experiment=False, verbose=False)
                        st.dataframe(reg_pull())
                        best_model = reg_compare()
                        st.dataframe(reg_pull())
                        reg_save(best_model, 'best_model')

                    st.success("Model training complete. You can download the trained model.")
                except ValueError as e:
                    st.error(f"Error in model setup: {str(e)}")
    else:
        st.warning("Please upload a dataset first.")

# Model download option
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model available. Please run modelling first.")