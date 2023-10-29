# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:24:11 2023

@author: keshav
"""


import pickle
import numpy as np
import streamlit as st


#load the saved model
final_model=pickle.load(open("C:/HOPE/A I/Machine Learning/MachineLearningClassification/finalmodelClassification.sav","rb"))

#creating a finction for prediction

def purchase_prediction(input_data):

    input_data_as_numpy_array=np.asarray(input_data)
    
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    
    Prediction=final_model.predict(input_data_reshaped)
    
    if(Prediction==1):
        return'They will purchase it'
    else:
        return'They will not purchase it'
                
        
def main():
    
# giving a title for webpage
    st.title('purchase Prediction web app')
    
#getting input data from the user

    userID=st.text_input("Enter User ID: ")
    Age= st.text_input("Enter Age: ")
    Estimated_Salary=st.text_input("Salary: ")
    Gender=st.text_input("Enter Gender Male 0 or 1 : ")
    
    #code for prediction
    pred = ""
    
    #creating a button for prediction
    if st.button('Purchase Prediction'):
        pred = purchase_prediction([userID,Age,Estimated_Salary,Gender])

    
    st.success(pred)
    
    
    
if __name__ == '__main__':
    main()

    

