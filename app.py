import streamlit as st
import pandas as pd
import numpy as np 
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


st.title('Medical Diagnostic Web App ⚕️')
st.subheader("Does the patient have diabetes?")
df = pd.read_csv('diabetes.csv')
if st.sidebar.checkbox('View data',False):
    st.write(df)
if st.sidebar.checkbox('View Distribution', False):
    df.hist()
    plt.tight_layout()
    st.pyplot()
    
#step 1:
model = open('rfc.pickle', 'rb')
clf = pickle.load(model)
model.close()

#step 2: get user input
pregs = st.number_input('Pregnancies',0,20,0)
plas = st.slider('Glucose', 40,200,40)
pres = st.slider('BloodPressure', 20,150,20)
skin = st.slider('SkinThickness',7,99,7)
insulin = st.slider('Insulin',14,850,14)
bmi = st.slider('BMI', 18,70,18)
dpf = st.slider('DiabetesPedigreeFunction',0.05,2.50, 0.05)
age = st.slider('Age', 21,90,21)

#step 3: user input to models input

input_data = [[pregs, plas, pres, skin, insulin, bmi, dpf, age]]

#step 4: get prediction and print results

prediction = clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction == 0 :
        st.subheader("Non diabetic")
    else:
        st.subheader("Diabetic")
