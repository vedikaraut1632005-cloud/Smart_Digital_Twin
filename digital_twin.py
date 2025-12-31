import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time

st.title("Smart Digital Twin for Predictive Maintenance")
st.write("This demo shows how a digital twin can monitor machine health in real-time.")

# Function to create fake sensor data
def generate_sensor_data(fault=False):
    temp = 40 + np.random.randn() * 2
    vib = 2 + np.random.randn() * 0.5
    curr = 10 + np.random.randn() * 1
    if fault:
        temp += np.random.uniform(10, 15)
        vib += np.random.uniform(2, 4)
    return temp, vib, curr

# Empty data storage
data = pd.DataFrame(columns=["Temperature", "Vibration", "Current"])

# Step 1: Train model with normal data
for i in range(50):
    temp, vib, curr = generate_sensor_data()
    data.loc[len(data)] = [temp, vib, curr]

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(data)

# Placeholder for live updates
placeholder = st.empty()
st.subheader("Live Machine Data Monitoring")

# Step 2: Simulate real-time readings
for i in range(200):
    fault = i > 120  # Fault appears after 120 readings
    temp, vib, curr = generate_sensor_data(fault)
    data.loc[len(data)] = [temp, vib, curr]
    prediction = model.predict([[temp, vib, curr]])[0]

    # Display on dashboard
    with placeholder.container():
        st.line_chart(data)
        if prediction == -1:
            st.error("Fault Detected! Machine behavior abnormal.")
        else:
            st.success("Machine running normally.")
        st.write(f"Temperature: {temp:.2f} °C | Vibration: {vib:.2f} | Current: {curr:.2f} A")
    time.sleep(0.3)
