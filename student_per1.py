import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_model():
    with open ("student_final_model1.pkl", 'rb') as file:
        model, scaler, le=pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data, scaler,le):
    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']])[0]
    df=pd.DataFrame([data])
    df_transformed= scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler,le=load_model()
    processed_data=preprocessing_input_data(data,scaler,le)
    prediction=model.predict(processed_data)
    return prediction

def main():
    st.title("Student performance prediction")
    st.write("enter your data to get prediction")

    hour_studied=st.number_input('Hours studied', min_value=1, max_value=10, value=5)
    prev_score=st.number_input('Previous Scores', min_value=40, max_value=100, value=40)
    exa=st.selectbox('Extracurricular Activities', ["Yes", "No"])
    sleep_hrs=st.number_input('Sleep Hours', min_value=1, max_value=19, value=8)
    no_of_paper_solved=st.number_input('Sample Question Papers Practiced ', min_value=1, max_value=20, value=3)

    if st.button("Predict your performance"):
        user_data= {
            'Hours Studied':hour_studied,
            'Previous Scores':prev_score,
            'Extracurricular Activities':exa,
            'Sleep Hours':sleep_hrs,
            'Sample Question Papers Practiced':no_of_paper_solved
        }
        prediction=predict_data(user_data)
        st.success(f'Your performance result is {prediction}')



if __name__=="__main__":
    main()