# AutoML Task Generalizer

## Overview
AutoML Task Generalizer is a web-based application built using Streamlit that automates the process of training machine learning models for both classification and regression tasks. The app offers a user-friendly interface for data upload, exploratory data analysis (EDA), model training, and downloading trained models. 

## Features
- **Data Upload**: Upload your CSV dataset for processing.
- **Exploratory Data Analysis**: Generate a profiling report of the dataset to understand its structure and statistics.
- **Automated Model Training**: Choose a target column, and the app automatically determines the task type (classification or regression) and builds the best model using PyCaret.
- **Download Trained Model**: Download the trained model as a `.pkl` file for deployment or further use.

## How to Use
### 1. Installation
Ensure you have Python installed (version 3.7+ recommended). Clone the repository and install the required packages:
```bash
git clone <AutoPredict_V1>
cd automl-task-generalizer
pip install -r requirements.txt
```

### 2. Run the App
Launch the Streamlit app using the following command:
```bash
streamlit run app.py
```

### 3. Usage
- **Upload**: Navigate to the "Upload" tab and upload your dataset in CSV format. The dataset will be displayed and saved locally as `dataset.csv`.
- **Profiling**: Visit the "Profiling" tab to generate an EDA report. You can download this report as `profile_report.html`.
- **Modelling**:
  - Select the "Modelling" tab and choose the target column from your dataset.
  - The app will detect the task type (classification or regression) based on the target's data type.
  - Click on "Run Modelling" to train models using PyCaret and view the setup and comparison results.
  - The best model will be saved locally as `best_model.pkl`.
- **Download**: Go to the "Download" tab to download the trained model for further use.

## Project Structure
```
.
├── app.py                # Main script for the Streamlit app
├── dataset.csv           # Uploaded dataset (created after data upload)
├── profile_report.html   # Generated profiling report (created after profiling)
├── best_model.pkl        # Saved trained model (created after modelling)
├── requirements.txt      # Required Python packages
└── README.md             # Project README file
```

## Dependencies
- **Python 3.7+**
- **Streamlit**: For creating the web app interface.
- **Pandas**: For data manipulation.
- **PyCaret**: For automated machine learning (classification and regression).
- **YData Profiling**: For generating exploratory data analysis reports.

Install these dependencies via:
```bash
pip install -r requirements.txt
```

### Sample Requirements File (`requirements.txt`)
```
streamlit
pandas
pycaret
ydata-profiling
```

## Acknowledgements
This project leverages the power of Streamlit for building interactive web applications and PyCaret for simplifying the machine learning model building process. YData Profiling is used to replace `pandas-profiling` for efficient EDA.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contact
For any questions or suggestions, please contact me at noecaremee@gmail.com.
```
