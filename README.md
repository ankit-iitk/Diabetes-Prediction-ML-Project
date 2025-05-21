#  Diabetes Prediction using Machine Learning

This repository contains a machine learning project that predicts whether a person has diabetes based on diagnostic measurements. The project uses the Pima Indians Diabetes Dataset and includes data preprocessing, exploratory data analysis, model training, evaluation, and deployment.

##  Project Overview

Early detection of diabetes can help prevent severe complications. This project uses supervised learning algorithms to classify whether or not a patient has diabetes.

##  Features

- Cleaned and preprocessed dataset (Pima Indians Diabetes Dataset)
- Exploratory Data Analysis (EDA)
- Multiple ML models trained and evaluated (Logistic Regression, Random Forest, SVM, etc.)
- Performance metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix 
- Model deployment using a simple web interface (optional: Streamlit/Flask)

##  Project Structure

diabetes-prediction-ml/
│
├── data/
│ └── diabetes.csv
├── notebooks/
│ └── ML_Project_Diabetes_Prediction.ipynb
├── models/
│ └── final_diabetes_model.pkl
├── app/
│ └── app.py (Optional - for deployment)
├── requirements.txt
├── README.md
└── .gitignore


##  Tech Stack

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit or Flask (optional deployment)


The dataset includes features like:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (1 = Diabetic, 0 = Non-diabetic)

##  How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/ankit-iitk/Diabetes-Prediction-ML-Project.git
   cd diabetes-prediction-ml
2. Create a virtual environment
   python -m venv venv
   source venv/bin/activate
3. Install dependencies
   pip install -r requirements.txt
4. Run the Jupyter Notebook
   jupyter notebook notebooks/ML_Project_Diabetes_Prediction.ipynb
5. Run the web app
   streamlit run app.py


