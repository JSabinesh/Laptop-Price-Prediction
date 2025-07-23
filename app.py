import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

# Load the entire dataset
sample = pd.read_csv('laptops.csv')

st.title('Laptop Price Predictor')
st.write('Enter the laptop specifications to predict the price:')

# Define valid options for each field
VALID_STATUS = ['New', 'Refurbished']
VALID_STORAGE_TYPE = ['SSD', 'HDD', 'eMMC']
VALID_TOUCH = ['No', 'Yes']
VALID_BRANDS = ['Asus', 'HP', 'Lenovo', 'MSI', 'Acer', 'Apple', 'Dell', 'Medion', 'Alurin', 'Gigabyte', 'Razer', 'LG']

# Helper to get unique sorted values, filtered by valid list
get_valid_options = lambda col, valid: sorted(set(x for x in sample[col].dropna().unique() if x in valid)) or valid

status = st.selectbox('Status', VALID_STATUS)
brand = st.selectbox('Brand', get_valid_options('Brand', VALID_BRANDS))

# Filter models based on selected brand, only show models that are not anomalies
filtered_models = sample[sample['Brand'] == brand]['Model'].dropna().unique()
model_options = sorted([m for m in filtered_models if isinstance(m, str) and m.strip() and m.lower() not in [b.lower() for b in VALID_BRANDS + VALID_STATUS]])
if not model_options:
    model_options = ['']
model_name = st.selectbox('Model', model_options)

# CPU and GPU: filter out anomalies by requiring 'Intel', 'AMD', 'Apple', 'Qualcomm', or 'NVIDIA' in the name
cpu_options = sorted(set(x for x in sample['CPU'].dropna().unique() if any(kw in str(x) for kw in ['Intel', 'AMD', 'Apple', 'Qualcomm'])))
# Remove unwanted CPUs
unwanted_cpus = {'Intel Core i5-1235U', 'AMD 3015Ce', 'AMD 3015e', 'AMD 3020e'}
cpu_options = [cpu for cpu in cpu_options if cpu.strip() not in unwanted_cpus]
cpu = st.selectbox('CPU', cpu_options)

gpu_options = sorted(set(x for x in sample['GPU'].dropna().unique() if x and (any(kw in str(x) for kw in ['RTX', 'GTX', 'Radeon', 'NVIDIA', '610', 'Apple', 'Intel', 'AMD']))))
# Ensure 'None' is always the first option
if 'None' not in gpu_options:
    gpu_options = ['None'] + gpu_options
else:
    gpu_options = ['None'] + [g for g in gpu_options if g != 'None']
gpu = st.selectbox('GPU', gpu_options)

ram = st.number_input('RAM (GB)', min_value=2, max_value=64, value=8, step=2)
storage = st.number_input('Storage (GB)', min_value=32, max_value=2048, value=512, step=32)
storage_type = st.selectbox('Storage Type', VALID_STORAGE_TYPE)
screen = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)
touch = st.selectbox('Touch', VALID_TOUCH)

USD_TO_INR = 83  # Example conversion rate

if st.button('Predict Price'):
    # Prepare input for prediction (order must match training)
    input_data = pd.DataFrame({
        'Status': [status],
        'Brand': [brand],
        'Model': [model_name],
        'CPU': [cpu],
        'RAM': [ram],
        'Storage': [storage],
        'Storage type': [storage_type],
        'GPU': [gpu],
        'Screen': [screen],
        'Touch': [touch],
    })
    predicted_price_usd = model.predict(input_data)[0]
    predicted_price_inr = predicted_price_usd * USD_TO_INR
    st.success(f'Estimated Price: ₹{predicted_price_inr:,.0f}') 