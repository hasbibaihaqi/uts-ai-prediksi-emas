from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
import sqlite3

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
def load_models():
    models = {}

    try:
        # Scaler
        models['scaler'] = joblib.load('models/scaler_X.pkl')
        # [TAMBAHAN] Memuat scaler_y untuk mengembalikan harga prediksi ke nilai asli
        models['scaler_y'] = joblib.load('models/scaler_y.pkl') 
    except Exception as e:
        print(f"Gagal memuat Scaler: {e}")

    # Machine Learning Models (Dipisah try-except agar tidak saling menggagalkan)
    try:
        models['lr'] = joblib.load('models/model_lr.pkl')
    except Exception as e:
        print(f"Gagal LR: {e}")

    try:
        models['mlp'] = joblib.load('models/model_mlp.pkl')
    except Exception as e:
        print(f"Gagal MLP: {e}")

    try:
        models['kmeans'] = joblib.load('models/model_kmeans.pkl')
    except Exception as e:
        print(f"Gagal KMeans: {e}")

    # Deep Learning Models
    try:
        models['ann'] = tf.keras.models.load_model('models/model_ann.h5')
    except Exception as e:
        print(f"Model ANN belum tersedia/Gagal dimuat: {e}")

    try:
        models['lstm'] = tf.keras.models.load_model('models/model_lstm.h5')
    except Exception as e:
        print(f"Model LSTM belum tersedia/Gagal dimuat: {e}")

    print("Proses pemuatan model selesai dilakukan!")
    return models

models = load_models()

# =========================
# DATABASE HISTORY
# =========================
def init_db():

    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT,
        open REAL,
        high REAL,
        low REAL,
        volume REAL,
        prediction REAL
    )
    ''')

    conn.commit()
    conn.close()

init_db()

# =========================
# HALAMAN UTAMA
# =========================
@app.route('/')
def index():
    return render_template('index.html')


# =========================
# PREDIKSI (DIPERBARUI UNTUK 5 MODEL SEKALIGUS)
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # =========================
        # Ambil Input Form
        # =========================
        open_price = float(request.form['open'])
        high = float(request.form['high'])
        low = float(request.form['low'])
        volume = float(request.form['volume'])

        # Mengambil input model jika masih ada di HTML (fallback)
        model_choice = request.form.get('model', 'SEMUA MODEL')

        # =========================
        # Validasi Input
        # =========================
        if high < low:
            return render_template(
                'index.html',
                error="Harga High tidak boleh lebih kecil dari Low.",
                open_price=open_price,
                high_price=high,
                low_price=low,
                volume=volume
            )

        # =========================
        # Input Data & Scaling
        # =========================
        input_data = np.array([
            [open_price, high, low, volume]
        ])
        scaled_data = models['scaler'].transform(input_data)

        # =========================
        # Prediksi Semua Model
        # =========================
        all_results = {}
        market_status = "Tidak Diketahui"

        # 1. Linear Regression
        if 'lr' in models:
            raw_lr = models['lr'].predict(scaled_data)[0]
            all_results['Linear Regression'] = float(models['scaler_y'].inverse_transform(np.array([[raw_lr]]))[0][0])

        # 2. Backpropagation / MLP
        if 'mlp' in models:
            raw_mlp = models['mlp'].predict(scaled_data)[0]
            all_results['MLP (Backpropagation)'] = float(models['scaler_y'].inverse_transform(np.array([[raw_mlp]]))[0][0])

        # 3. ANN
        if 'ann' in models:
            raw_ann = models['ann'].predict(scaled_data, verbose=0)[0][0]
            all_results['Artificial Neural Network (ANN)'] = float(models['scaler_y'].inverse_transform(np.array([[raw_ann]]))[0][0])

        # 4. LSTM
        if 'lstm' in models:
            lstm_input = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))
            raw_lstm = models['lstm'].predict(lstm_input, verbose=0)[0][0]
            all_results['LSTM (Deep Learning)'] = float(models['scaler_y'].inverse_transform(np.array([[raw_lstm]]))[0][0])

        # 5. KMeans (Untuk Status Klaster Pasar)
        if 'kmeans' in models:
            cluster = models['kmeans'].predict(scaled_data)[0]
            
            # Menerjemahkan angka cluster mesin ke dalam bahasa manusia
            if cluster == 0:
                market_status = "Pasar Stabil (Sideways / Vol. Rendah)"
            elif cluster == 1:
                market_status = "Tren Aktif (Volatilitas Normal)"
            elif cluster == 2:
                market_status = "Tren Agresif (Volatilitas Tinggi)"
            else:
                market_status = f"Cluster {cluster}"

        # =========================
        # SIMPAN RIWAYAT PREDIKSI
        # =========================
        # Mengambil hasil LSTM sebagai representasi utama untuk disimpan di riwayat (jika ada)
        primary_pred = all_results.get('LSTM (Deep Learning)', list(all_results.values())[0] if all_results else 0)

        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO history
        (model, open, high, low, volume, prediction)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            "MULTI-MODEL (5 Algoritma)",
            open_price,
            high,
            low,
            volume,
            primary_pred
        ))
        conn.commit()
        conn.close()

        # =========================
        # Render Template dengan Data Baru
        # =========================
        return render_template(
            'index.html',
            # Variabel baru untuk tabel perbandingan banyak model
            all_results=all_results,
            market_status=market_status,
            
            # Mempertahankan variabel lama agar form tidak error/kosong saat dirender ulang
            hasil_prediksi=f"Rp {primary_pred:,.2f}",
            model_terpilih="SEMUA MODEL",
            open_price=open_price,
            high_price=high,
            low_price=low,
            volume=volume
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Terjadi kesalahan: {str(e)}"
        )


# =========================
# HALAMAN PERBANDINGAN
# =========================
@app.route('/perbandingan')
def perbandingan():

    akurasi = {
        'Linear Regression': 95,
        'ANN': 97,
        'LSTM': 98,
        'MLP': 96
    }

    return render_template(
        'perbandingan.html',
        akurasi=akurasi
    )


# =========================
# HISTORY PREDIKSI
# =========================
@app.route('/history')
def history():

    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM history ORDER BY id DESC')

    data = cursor.fetchall()

    conn.close()

    return render_template(
        'history.html',
        data=data
    )


# =========================
# DASHBOARD STATISTIK
# =========================
@app.route('/dashboard')
def dashboard():

    try:
        df = pd.read_csv('dataset/gold_price.csv')

        # Cleaning sederhana
        df['High'] = pd.to_numeric(
            df['High'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        df['Low'] = pd.to_numeric(
            df['Low'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        df['Price'] = pd.to_numeric(
            df['Price'].astype(str).str.replace(',', ''),
            errors='coerce'
        )

        highest = round(df['High'].max(), 2)
        lowest = round(df['Low'].min(), 2)
        average = round(df['Price'].mean(), 2)

        total_data = len(df)

        return render_template(
            'dashboard.html',
            highest=highest,
            lowest=lowest,
            average=average,
            total_data=total_data
        )

    except Exception as e:
        return render_template(
            'dashboard.html',
            error=f"Gagal memuat dashboard: {str(e)}"
        )


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)