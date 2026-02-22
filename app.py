import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Hospital Predictor", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title Section
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🏥 Hospital Waiting Time Predictor</h1>
    <p style='text-align: center;'>Predict patient waiting time using Machine Learning</p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app predicts hospital waiting time based on real-time factors like patients, doctors, and emergencies."
)

# Input Section
st.subheader("Enter Hospital Data")

col1, col2 = st.columns(2)

with col1:
    patients = st.slider("👥 Patients Waiting", 0, 100)
    doctors = st.slider("👨‍⚕️ Doctors Available", 1, 20)

with col2:
    emergency = st.slider("🚨 Emergency Cases", 0, 20)
    hour = st.slider("⏰ Hour of Day", 0, 23)

day = st.selectbox("📅 Day of Week", 
                   ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

day_map = {
    "Monday":0, "Tuesday":1, "Wednesday":2,
    "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6
}

st.divider()

# Prediction
if st.button("Predict Waiting Time"):
    input_data = [[patients, doctors, emergency, hour, day_map[day]]]
    result = model.predict(input_data)

    st.success(f"⏳ Estimated Waiting Time: {result[0]:.2f} minutes")

    if result[0] > 30:
        st.warning("⚠️ High waiting time expected!")
    else:
        st.info("✅ Waiting time is manageable.")
import matplotlib.pyplot as plt

if st.button("Show Impact of Patients"):
    patients_range = list(range(0, 100))
    predictions = []

    for p in patients_range:
        pred = model.predict([[p, doctors, emergency, hour, day_map[day]]])
        predictions.append(pred[0])

    plt.plot(patients_range, predictions)
    plt.xlabel("Patients")
    plt.ylabel("Waiting Time")
    st.pyplot(plt)