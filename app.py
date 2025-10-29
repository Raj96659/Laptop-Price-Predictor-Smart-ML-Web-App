import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💰",
    layout="centered",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #e0f7fa, #ffffff);
            font-family: 'Poppins', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 38px;
            font-weight: 700;
            color: #00796b;
            padding-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .stButton>button {
            background: linear-gradient(to right, #00796b, #26a69a);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 25px;
            font-size: 18px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #004d40, #00796b);
            transform: scale(1.03);
        }
        .prediction-card {
            background-color: #f0fdfa;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        h3 {
            color: #004d40;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATA ---
pipe = pickle.load(open('pipe1.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# --- HEADER ---
st.markdown("<h1 class='main-title'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict your laptop’s price using machine learning</p>", unsafe_allow_html=True)

# --- INPUT SECTIONS ---
with st.expander("🖥️ Basic Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox('Brand', df['Company'].unique())
        type_name = st.selectbox('Type', df['TypeName'].unique())
        os = st.selectbox('Operating System', df['OpSys'].unique())
        weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)
    with col2:
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
        touchscreen = st.radio('Touchscreen', ['No', 'Yes'])
        ips = st.radio('IPS Display', ['No', 'Yes'])

with st.expander("📺 Display Configuration"):
    col3, col4 = st.columns(2)
    with col3:
        screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
    with col4:
        resolution = st.selectbox('Screen Resolution', [
            '1920x1080', '1366x768', '1600x900', '3840x2160',
            '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
        ])

with st.expander("⚙️ Performance & Storage"):
    col5, col6 = st.columns(2)
    with col5:
        cpu = st.selectbox('CPU', df['Cpu Name'].unique())
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    with col6:
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
        gpu = st.selectbox('GPU', df['Gpu_Brand'].unique())

# --- PREDICT BUTTON ---
if st.button('🚀 Predict Price'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create query DataFrame
    query = pd.DataFrame([[
        company, type_name, ram, weight, touchscreen, ips, ppi, 
        cpu, hdd, ssd, gpu, os
    ]], columns=[
        'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi',
        'Cpu Name', 'HDD', 'SSD', 'Gpu_Brand', 'OpSys'
    ])

    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # --- OUTPUT CARD ---
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3>💰 Estimated Price: ₹ {predicted_price:,}</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray'>Note: This is an approximate predicted price based on ML model.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
