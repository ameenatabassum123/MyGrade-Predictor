import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# If you have a logo, save in your project folder and use it:
# st.image("logo.png", width=120)
st.title("🎓 MyGrade Predictor App ")
st.title("Predicts Student Final Exam Score")


st.markdown("""
Welcome! This interactive app predicts a student's final exam score based on their study habits and grades.Made for students to track the self progress.
- Enter details in the sidebar.
- Click 'Predict Final Exam Score' to view your result.
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

else:
    st.info("Enter the details and click the button to predict.")


y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
with st.expander("Model Evaluation Metrics & Details"):
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write("Feature Coefficients:")
    coef_df = pd.DataFrame(model.coef_, features, columns=["Coefficient"])
    st.dataframe(coef_df)


st.markdown("### Actual vs Predicted Exam Scores (Test Data)")
fig, ax = plt.subplots(figsize=(5,3))
ax.scatter(y_test, y_pred_test, alpha=0.7)
ax.plot([0,20],[0,20],'--',color='red')
ax.set_xlabel("Actual  Score")
ax.set_ylabel("Predicted  Score")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)




if st.checkbox("Show Sample Data Table"):
    st.dataframe(df.head())


st.markdown("---")
st.markdown("Made by [Ameena Tabassum]. Contact: ameenatabassum1664@gmail.com")


