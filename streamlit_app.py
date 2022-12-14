import pickle
import streamlit as st

import numpy as np
import pandas as pd

# loading the trained model
pickle_in = open('regmodel.pkl','rb')
regmodel = pickle.load(pickle_in)
# loading the scaler
scaler=pickle.load(open('scaling.pkl','rb'))
@st.cache()

#Define the function making the prediction using the data inputted by users

def prediction(data):
    data = scaler.transform(data)
    return regmodel.predict([data])

#This is the main function in which we define our webpage
def main():
    
    st.title("Housing prediction")
    html_temp = ""

    st.markdown(html_temp, unsafe_allow_html=True)

    crim = st.number_input("CRIM")
    zn   = st.number_input("ZN")
    indus= st.number_input("INDUS")
    nox  = st.number_input("NOX")
    rm   = st.number_input("RM")
    age  = st.number_input("Age")
    dis  = st.number_input("DIS")
    rad  = st.number_input("RAD")
    tax  = st.number_input("TAX")
    ptra = st.number_input("PTRATIO")
    b    = st.number_input("B")
    lstat = st.number_input("LSTAT")

    data = [crim, zn, indus, nox, rm, age, dis, rad, tax, ptra, b, lstat]
    if st.button("Predict"):
        result = prediction(data)
        st.success(f"The price is {result}")

if __name__=='__main__': 
    main()