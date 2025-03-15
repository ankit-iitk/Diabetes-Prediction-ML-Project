import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split


model = pickle.load(open('model.pkl','rb')) 
  
df = pd.read_csv('diabetes.csv')

x = df.drop('Outcome',axis = 1)
y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

rfe = RFE(RandomForestClassifier(n_estimators = 100,criterion = 'gini',max_depth = 10),n_features_to_select=4)

rfe.fit(x_train,y_train)

x_train = rfe.fit_transform(x_train,y_train)
x_test = rfe.transform(x_test)

st.title('Diabetes Prediction Using ML')
st.write("")

st.sidebar.title("Diabetes Prediction")
st.sidebar.write("")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file",type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    st.write(f"DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")


preg = st.number_input("Number of Pregnancies")
st.write(f'Number of pregnancies : {preg}')
st.write("")

gluc_level = st.number_input("Glucose Level") 
st.write(f'Your Glucose Level is : {gluc_level}') 
st.write("")

bp = st.number_input("Blood Pressure Level")
st.write(f'Your Blood Pressure Level is : {bp}')
st.write("")

stv = st.number_input("Skin Thickness Value")
st.write(f'Your Skin Thickness Value is : {stv}')
st.write("")

insulin = st.number_input("Insulin Value")
st.write(f'Your Insulin Value is : {insulin}')
st.write("")

bmi = st.number_input("BMI Value")
st.write(f'Your BMI Value is : {bmi}')
st.write("")

dpf = st.number_input("Diabetes Pedigree Function Value")
st.write(f'Your Diabetes Pedigree Function Value is : {dpf}')
st.write("")

age = st.number_input("Age of Person")
st.write(f"Your age is : {age}") 
st.write("")



def provide_health_tips(prediction):
    if prediction[0] == 1:
        st.subheader("Health Tips for Diabetic Management")
        st.write("""
            - Monitor your blood sugar regularly.
            - Follow a balanced, low-carb diet.
            - Stay physically active: at least 30 minutes of moderate activity daily.
            - Manage stress through relaxation techniques.
        """)
    else:
        st.subheader("Health Tips for Prevention")
        st.write("""
            - Maintain a healthy weight.
            - Stay active to improve insulin sensitivity.
            - Follow a balanced diet and avoid sugary foods.
            - Get regular health check-ups.
        """)


if st.button("Predict"):
    input_df = pd.DataFrame({
        'Pregnancies':[preg],
        'Glucose Level':[gluc_level],
        'BP':[bp],
        'stv':[stv],
        'insulin':[insulin],
        'bmi':[bmi],
        'dpf':[dpf],
        'age':[age]
    })
    st.table(input_df)
    input_data = np.array([preg, gluc_level, bp, stv, insulin, bmi, dpf, age])
  
    input_data = input_data.reshape(1, -1)
  
    scaled_data = scaler.transform(input_data) 
  
    x_transformed = rfe.transform(scaled_data)  
    prediction = model.predict(x_transformed)

    

    if prediction[0] == 1:
        # Diabetic prediction - Red Button
        st.markdown(
            """
            <style>
            .diabetic-button {
                background-color: #ff4d4d;
                color: white;
                padding: 10px;
                font-size: 20px;
                border-radius: 10px;
                text-align: center;
            }
            </style>
            <div class="diabetic-button">
                The Person is Diabetic
            </div>
            """, unsafe_allow_html=True
        )
        st.warning("You are at a higher risk of diabetes! Please consult with a healthcare provider.")
        provide_health_tips(prediction)

    else:
        # Non-Diabetic prediction - Green Button
        st.markdown(
            """
            <style>
            .non-diabetic-button {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 20px;
                border-radius: 10px;
                text-align: center;
            }
            </style>
            <div class="non-diabetic-button">
                The person is Non Diabetic
            </div>
            """, unsafe_allow_html=True
        )
        st.success("Great job! You are at low risk of diabetes. Keep maintaining a healthy lifestyle.")
        
        provide_health_tips(prediction)

st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f9;
        color: #1e2a47;
    }
    .stButton>button {
        background-color: #00796b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)


