import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
def preprocess_data(df):
    df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)
def calculate_bmi(weight, height):
    if height == 0:
        return 0
    height_in_meters = height / 100
    bmi = weight / (height_in_meters ** 2)
    return bmi
def main():
    st.title("Medical Insurance Prediction Model")
    st.sidebar.header('USER INPUT PARAMETERS')
    p1 = st.sidebar.slider("Enter Your Age", 18, 100)
    s1 = st.sidebar.selectbox("Sex", ["Male", "Female"])
    p2 = 1 if s1 == "Male" else 0
    weight = st.sidebar.number_input("Enter Your Weight (kg)", min_value=0.0, max_value=200.0)
    height_feet = st.sidebar.number_input("Enter Your Height (feet)", min_value=0.0, max_value=8.0)
    height_cm = height_feet * 30.48  
    calculated_bmi = calculate_bmi(weight, height_cm)
    p3_placeholder = st.sidebar.empty()
    p3_placeholder.text(f"Calculated BMI: {round(calculated_bmi, 2)}")
    p3 = st.sidebar.number_input("this is Your BMI Value", float(calculated_bmi), float(medical_df['bmi'].max()))
    p4 = st.sidebar.slider("Enter Number of Children", 0, 5)
    s2 = st.sidebar.selectbox("Smoker", ["Yes", "No"])
    p5 = 1 if s2 == "Yes" else 0
    region_mapping = {0: 'Southeast', 1: 'Southwest', 2: 'Northwest', 3: 'Northeast'}
    region_options = list(region_mapping.values())  
    p6 = st.sidebar.selectbox("Enter Your Region", region_options)
    if st.sidebar.button('Predict'):
        p6_numeric = list(region_mapping.keys())[list(region_mapping.values()).index(p6)]
        input_data = np.array([p1, p2, calculated_bmi, p4, p5, p6_numeric]).reshape(1, -1)
        prediction = lg.predict(input_data)
        st.balloons()
        st.success(f'Medical Insurance prediction : {round(prediction[0], 2)}')
if __name__ == '__main__':
    csv_file_path = 'insurance.csv'  
    medical_df = pd.read_csv(csv_file_path)
    preprocess_data(medical_df)
    X = medical_df.drop('charges', axis=1)
    y = medical_df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
    lg = LinearRegression()
    lg.fit(X_train, y_train)
    main()
