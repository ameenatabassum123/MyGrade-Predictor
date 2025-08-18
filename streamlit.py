import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error




st.title("🎓 MyGrade Predictor App")
st.title("Predicts Student Final Exam Score")



st.markdown("""
Welcome! This interactive app predicts a student's final exam score based on their study habits and grades. Made for students to track self progress.
- Enter details in the sidebar.
- Click 'Predict Final Exam Score' to view your result and visual insights.
""")



df = pd.read_csv('student-por.csv')
df['activities'] = df['activities'].map({'yes': 1, 'no': 0})

features = ['studytime', 'absences', 'activities', 'G1', 'G2']
X = df[features]
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)



st.sidebar.header("Input Student Data")
studytime = st.sidebar.slider('Study Time (1–4)', 1, 4, 2)
absences = st.sidebar.slider('Absences', int(df.absences.min()), int(df.absences.max()), 0)
activities = st.sidebar.radio('Participates in Activities?', ['Yes', 'No'])
G1 = st.sidebar.slider('First Period Grade (0–20)', int(df.G1.min()), int(df.G1.max()), 10)
G2 = st.sidebar.slider('Second Period Grade (0–20)', int(df.G2.min()), int(df.G2.max()), 10)

input_data = pd.DataFrame({
    'studytime': [studytime],
    'absences': [absences],
    'activities': [1 if activities == 'Yes' else 0],
    'G1': [G1],
    'G2': [G2]
})



if st.button('Predict Final Exam Score'):
    predicted_score = model.predict(input_data)[0]
    st.markdown(f"### Predicted Final Exam Score : {predicted_score:.2f}")

    
    
    st.markdown("#### Your Grade Progression")
    fig1, ax1 = plt.subplots()
    ax1.plot(['G1', 'G2', 'Predicted G3'], [G1, G2, predicted_score], marker='o', color='blue')
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 20)
    st.pyplot(fig1)

   
    
    st.markdown("#### Distribution of Final Exam Scores (You Highlighted)")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['G3'], bins=20, color='skyblue', edgecolor='black')
    ax2.axvline(predicted_score, color='red', linestyle='dashed', linewidth=2,
                label=f'Your Predicted Score ({predicted_score:.2f})')
    ax2.set_xlabel("Final Exam Score")
    ax2.set_ylabel("Number of Students")
    ax2.legend()
    st.pyplot(fig2)

   
    
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    with st.expander("Model Evaluation Metrics & Details"):
        st.write(f"**R² Score:** {r2:.2f}")
        st.write(f"**Mean Absolute Error:** {mae:.2f}")
        st.write("Feature Coefficients:")
        coef_df = pd.DataFrame(model.coef_, features, columns=["Coefficient"])
        st.dataframe(coef_df)


    if st.checkbox("Show Sample Data Table"):
        st.dataframe(df.head())

else:
    st.info("Enter the details and click the button to predict. Graphs will appear after prediction.")


st.markdown("---")
st.markdown("Made by [Ameena Tabassum]. Contact: ameenatabassum1664@gmail.com")
