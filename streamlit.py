import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----- 1. Set a light peach background -----
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF6E9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----- 2. App Title & Description -----
st.title("MyGrade Predictor")
st.markdown("""
Welcome!  
Estimate your final exam score by entering your study details.  
Click **Predict Final Exam Score** to see your personalized result and insights!
""")

# ----- 3. Load & Prepare Data -----
df = pd.read_csv('student-por.csv')
df['activities'] = df['activities'].map({'yes': 1, 'no': 0})
features = ['studytime', 'absences', 'activities', 'G1', 'G2']
X = df[features]
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ----- 4. Sidebar Inputs -----
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

# ----- 5. Prediction & Visualizations only after button -----
if st.button('Predict Final Exam Score'):
    predicted_score = model.predict(input_data)[0]
    st.markdown(f"### Predicted Final Exam Score : {predicted_score:.2f}")

    # Histogram of Final Exam Scores
    st.markdown("#### Distribution of Final Exam Scores")
    fig1, ax1 = plt.subplots()
    ax1.hist(df['G3'], bins=20, color='skyblue', edgecolor='black')
    ax1.set_xlabel("Final Exam Score")
    ax1.set_ylabel("Number of Students")
    st.pyplot(fig1)

    # Bar Chart: Average Scores by Activity Participation
    st.markdown("#### Average Scores by Activity Participation")
    avg_scores = df.groupby('activities')['G3'].mean()
    fig2, ax2 = plt.subplots()
    ax2.bar(['No', 'Yes'], avg_scores, color=['orange', 'green'])
    ax2.set_xlabel("Participation in Activities")
    ax2.set_ylabel("Average Final Exam Score")
    st.pyplot(fig2)

else:
    st.info("Enter your details and click the button to predict. Graphs will be shown after prediction.")

# ----- 6. Footer -----
st.markdown("---")
st.markdown("Made by [Your Name] • For students and learning!")
