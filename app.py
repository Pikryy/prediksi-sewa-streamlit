import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Prediksi Harga Sewa Properti")

# Membaca data dan melatih model 
@st.cache_data
def buat_model():
    df = pd.read_pickle('dataset_valencia.pkl')
    model = LinearRegression()
    # X = sale_price_eur_m2, Y = rent_price_eur_m2_month
    model.fit(df[['sale_price_eur_m2']], df['rent_price_eur_m2_month'])
    return model

model = buat_model()

# Input Pengguna
harga_beli = st.number_input("Masukkan Harga Beli (EUR/m²):", min_value=0, value=3000, step=100)

# Prediksi
if st.button("Hitung Prediksi"):
    # Format input ke dalam DataFrame agar sesuai dengan model
    input_data = pd.DataFrame({'sale_price_eur_m2': [harga_beli]})
    
    # Lakukan prediksi
    hasil = model.predict(input_data)[0]
    
    # Tampilkan hasil
    st.success(f"Perkiraan Harga Sewa: € {hasil:.2f} / m² / bulan")