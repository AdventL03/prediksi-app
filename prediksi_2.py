import streamlit as st
import datetime # Import library datetime
import pandas as pd
import datetime
import numpy as np
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
pd.set_option('display.max_rows',None,'display.max_columns',None,'display.max_colwidth',800)
# Import library lain yang dibutuhkan untuk pemrosesan data dan fungsi seperti speed_km, parse_time_to_seconds, filter_feature_aerobic

# --- Fungsi-fungsi Pemrosesan Data (Pastikan ada di sini atau di file terpisah yang diimpor) ---
# def speed_km(pace_str): ...
# def parse_time_to_seconds(time_str): ...
# def filter_feature_aerobic(filtered_df): ...
# Pastikan fungsi-fungsi ini sudah Anda copy-paste atau impor ke dalam script Streamlit Anda

# Muat model di awal script
try:
    # Sesuaikan nama file model jika berbeda
    with open('model_linreg.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("File model 'model_linreg.pkl' tidak ditemukan. Pastikan file model sudah diunggah atau berada di lokasi yang benar.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()


st.title("Prediksi Waktu Marathon Berdasarkan Data Garmin")

uploaded_file = st.file_uploader("Unggah file aktivitas Garmin Anda (.csv)")

target_date = st.date_input("Pilih Tanggal Target (Tanggal Lomba Marathon atau Tanggal Saat Ini)", datetime.date.today())

# Konversi tanggal input Streamlit menjadi objek datetime Pandas
target_date_pandas = pd.to_datetime(target_date)

# Inisialisasi session state untuk menyimpan hasil prediksi dan status tampilan strategi
if 'prediction_seconds' not in st.session_state:
    st.session_state['prediction_seconds'] = None
if 'show_pacing_strategy' not in st.session_state:
    st.session_state['show_pacing_strategy'] = False


# --- Bagian Pemrosesan File dan Prediksi ---
if uploaded_file is not None:
    df_user_raw = pd.read_csv(uploaded_file)

    # --- Kode Pemrosesan Data (Sama seperti sebelumnya) ---
    # Ini termasuk:
    # - Mengubah format tanggal (Activity Date atau Date)
    # - Memfilter 6 bulan sebelum target_date_pandas
    # - Menangani missing value ('--')
    # - Mengubah format waktu ke detik (Moving Time, Elapsed Time)
    # - Mengubah tipe data ke numerik
    # - Terapkan filter_feature_aerobic dan hitung persentase
    # - Agregasi data ke satu baris (aggregated_data_user)
    # - Ambil input Usia dan Berat Badan
    # - Pastikan nama dan urutan kolom sesuai dengan X_train

    # Contoh struktur minimal, sisipkan kode pemrosesan lengkap Anda di sini:
    try:
        # --- Mulai Sisipkan Kode Pemrosesan Data Anda di Sini ---
        # Mengubah kolom 'Activity Date' menjadi tipe data datetime
        if 'Activity Date' in df_user_raw.columns:
            df_user_raw['Date'] = pd.to_datetime(df_user_raw['Activity Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        elif 'Date' in df_user_raw.columns:
             df_user_raw['Date'] = pd.to_datetime(df_user_raw['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
        else:
             st.error("Tidak ditemukan kolom tanggal yang dikenali ('Activity Date' atau 'Date'). Harap periksa file CSV Anda.")
             st.session_state['prediction_seconds'] = None # Reset prediksi jika ada error
             st.session_state['show_pacing_strategy'] = False
             st.stop()

        # Menentukan tanggal awal (6 bulan sebelum tanggal target)
        start_date = target_date_pandas - pd.DateOffset(months=6)

        # Memfilter data berdasarkan tipe aktivitas dan rentang tanggal
        filtered_df_user = df_user_raw[
            (df_user_raw['Activity Type'].isin(['Running', 'Treadmill'])) &
            ((df_user_raw['Date'] >= start_date) & (df_user_raw['Date'] < target_date_pandas))
        ].copy()

        if filtered_df_user.empty:
            st.warning("Tidak ada data aktivitas Running/Treadmill dalam 6 bulan sebelum tanggal yang dipilih.")
            st.session_state['prediction_seconds'] = None # Reset prediksi
            st.session_state['show_pacing_strategy'] = False
            st.stop()

        # Identifikasi dan hapus baris dengan '--' di Avg Pace atau Best Pace
        rows_to_drop_user = filtered_df_user[(filtered_df_user['Avg Pace'] == '--') | (filtered_df_user['Best Pace'] == '--')].index
        filtered_df_user = filtered_df_user.drop(rows_to_drop_user).reset_index(drop=True)

        def speed_km(pace_str):
            pace_split = pace_str.str.split(':', expand=True).astype(float)
            pace_minutes = pace_split[0] + (pace_split[1] / 60)
            speed_km = 60 / pace_minutes
            return speed_km
        
        # Terapkan fungsi speed_km
        if 'Avg Pace' in filtered_df_user.columns and 'Best Pace' in filtered_df_user.columns:
            filtered_df_user['Average Speed'] = speed_km(filtered_df_user['Avg Pace'])
            filtered_df_user['Max Speed'] = speed_km(filtered_df_user['Best Pace'])
        else:
             st.warning("Kolom 'Avg Pace' atau 'Best Pace' tidak ditemukan. Kecepatan rata-rata dan maks tidak akan digunakan.")
             filtered_df_user['Average Speed'] = pd.NA
             filtered_df_user['Max Speed'] = pd.NA

        def parse_time_to_seconds(time_str):
            if isinstance(time_str, str):
                try:
                    # Coba format hh:mm:ss atau mm:ss
                    parts = time_str.split(':')
                    if len(parts) == 3: # hh:mm:ss
                        hours, minutes, seconds = map(float, parts)
                        return int(hours * 3600 + minutes * 60 + seconds)
                    elif len(parts) == 2: # mm:ss or mm:ss.decimal
                        # Pisahkan detik dan desimal jika ada
                        if '.' in parts[1]:
                            seconds_parts = parts[1].split('.')
                            seconds = float(seconds_parts[0]) + float('0.' + seconds_parts[1])
                        else:
                            seconds = float(parts[1])
                        minutes = float(parts[0])
                        return int(minutes * 60 + seconds)
                except ValueError:
                    return None
            return None
        
        # Terapkan parse_time_to_seconds
        if 'Elapsed Time' in filtered_df_user.columns and 'Moving Time' in filtered_df_user.columns:
             filtered_df_user['Elapsed Time'] = filtered_df_user['Elapsed Time'].apply(parse_time_to_seconds).astype('Int64')
             filtered_df_user['Moving Time'] = filtered_df_user['Moving Time'].apply(parse_time_to_seconds).astype('Int64')
        else:
             st.warning("Kolom 'Elapsed Time' atau 'Moving Time' tidak ditemukan. Fitur waktu tidak akan digunakan.")
             filtered_df_user['Elapsed Time'] = pd.NA
             filtered_df_user['Moving Time'] = pd.NA


        # Ubah tipe data ke numerik
        cols_to_numeric = ['Average Speed', 'Max Speed', 'Avg HR', 'Max HR',
                           'Avg Run Cadence', 'Total Ascent', 'Total Descent',
                           'Moving Time', 'Elapsed Time', 'Distance', 'Aerobic TE']
        for col in cols_to_numeric:
            if col in filtered_df_user.columns:
                 # Ganti '--' menjadi NaN sebelum konversi numerik jika belum dilakukan
                 filtered_df_user[col] = filtered_df_user[col].replace('--', pd.NA)
                 filtered_df_user[col] = pd.to_numeric(filtered_df_user[col], errors='coerce')
            else:
                 st.warning(f"Kolom '{col}' tidak ditemukan.")
                 # Tambahkan kolom dengan nilai NaN jika tidak ada, agar agregasi tidak error
                 filtered_df_user[col] = pd.NA

        def filter_feature_aerobic(filtered_df):
            if filtered_df['Aerobic TE'] <= 3.9:
                return 'Low Aerobic'
            elif (filtered_df['Aerobic TE'] >= 4) and (filtered_df['Aerobic TE'] < 5):
                return 'High Aerobic'
            elif filtered_df['Aerobic TE'] >= 5:
                return 'Anaerobic'
            else:
                None

        # Terapkan filter_feature_aerobic dan hitung persentase
        smy_user = filtered_df_user.copy()
        if 'Aerobic TE' in smy_user.columns:
            # Pastikan kolom 'Aerobic TE' bertipe numerik sebelum ini
            smy_user['Aerobic TE Category'] = smy_user['Aerobic TE'].apply(lambda x: filter_feature_aerobic({'Aerobic TE': x})) # Panggil fungsi dengan format yang sesuai
            aerobic_counts_user = smy_user['Aerobic TE Category'].value_counts()
            total_activities_user = aerobic_counts_user.sum()
        else:
             st.warning("Kolom 'Aerobic TE' tidak ditemukan. Fitur persentase Aerobic tidak akan digunakan.")
             aerobic_counts_user = pd.Series() # Buat series kosong jika kolom tidak ada
             total_activities_user = 0


        # Agregasi data user
        # Pastikan semua kolom yang ada di X_train dari notebook Anda diagregasi di sini
        aggregated_data_user = filtered_df_user.agg({
            'Date': 'count', # Jumlah aktivitas
            'Average Speed': 'mean',
            'Max Speed': 'max',
            'Avg HR': 'mean',
            'Max HR': 'max',
            'Avg Run Cadence': 'mean',
            'Total Ascent': 'sum',
            'Total Descent': 'sum',
            'Moving Time': 'sum',
            'Elapsed Time': 'sum',
            'Distance': 'sum', # Total jarak lari
        })

        aggregated_data_user['Max Distance'] = filtered_df_user['Distance'].max() # Jarak terpanjang

        # Tambahkan persentase Aerobic
        if total_activities_user > 0:
             aggregated_data_user['Low Aerobic (%)'] = (aerobic_counts_user.get('Low Aerobic', 0) / total_activities_user) * 100
             aggregated_data_user['High Aerobic (%)'] = (aerobic_counts_user.get('High Aerobic', 0) / total_activities_user) * 100
             aggregated_data_user['Anaerobic (%)'] = (aerobic_counts_user.get('Anaerobic', 0) / total_activities_user) * 100
        else:
             aggregated_data_user['Low Aerobic (%)'] = 0
             aggregated_data_user['High Aerobic (%)'] = 0
             aggregated_data_user['Anaerobic (%)'] = 0


        # Ambil input Usia dan Berat Badan
        st.subheader("Informasi Personal Tambahan")
        # Gunakan session_state untuk mempertahankan nilai input
        if 'user_age' not in st.session_state:
            st.session_state['user_age'] = 30
        if 'user_weight' not in st.session_state:
            st.session_state['user_weight'] = 50.0

        age = st.number_input("Masukkan Usia Anda", min_value=1, max_value=100, value=st.session_state['user_age'], key='age_input')
        weight = st.number_input("Masukkan Berat Badan Anda (kg)", min_value=1.0, max_value=200.0, value=st.session_state['user_weight'], key='weight_input')
        
        # Update session_state saat nilai berubah
        st.session_state['user_age'] = age
        st.session_state['user_weight'] = weight

        # Gender
        gender = st.radio("Pilih Gender Anda", ("Perempuan", "Laki-laki"))
        gender_value = 0 if gender == "Perempuan" else 1

        aggregated_data_user['Age'] = age
        aggregated_data_user['Weight'] = weight
        aggregated_data_user['Gender'] = gender_value

        # Pastikan aggregated_data_user adalah DataFrame dengan 1 baris dan nama kolom yang benar
        data_pengguna_agregat = pd.DataFrame(aggregated_data_user).T
        data_pengguna_agregat = data_pengguna_agregat.reset_index(drop=True)

        # Sesuaikan nama kolom agar sesuai dengan nama kolom di X_train (dari notebook)
        # Daftar nama kolom dari X di notebook (sesuaikan jika ada perubahan):
        expected_cols_mapping = {
            'Date': 'Activity', # Di X_train namanya "Activity", jumlah aktivitas
            'Average Speed': 'Average Speed (km/h)',
            'Max Speed': 'Max Speed (km/h)',
            'Avg HR': 'Avg HR',
            'Max HR': 'Max HR',
            'Avg Run Cadence': 'Avg Run Cadence',
            'Total Ascent': 'Total Ascent',
            'Total Descent': 'Total Descent',
            'Moving Time': 'Moving Time',
            'Elapsed Time': 'Elapsed Time',
            'Distance': 'Distance', # Total Distance
            'Max Distance': 'Max Distance', # Max Distance
            'Low Aerobic (%)': 'Low Aerobic (%)',
            'High Aerobic (%)': 'High Aerobic (%)',
            'Anaerobic (%)': 'Anaerobic (%)',
            'Age': 'Age',
            'Weight': 'Weight',
            'Gender': 'Gender'
        }

        # Ganti nama kolom
        data_pengguna_agregat = data_pengguna_agregat.rename(columns=expected_cols_mapping)

        # Pastikan hanya kolom yang diharapkan ada dan urutannya benar
        final_cols_order = list(expected_cols_mapping.values())
        # Hapus kolom yang tidak ada di data_pengguna_agregat sebelum reindex
        cols_to_keep = [col for col in final_cols_order if col in data_pengguna_agregat.columns]
        data_pengguna_agregat = data_pengguna_agregat[cols_to_keep]

        # Tambahkan kolom yang hilang jika ada dan isi dengan NaN (ini bisa terjadi jika file CSV user tidak lengkap)
        for col in final_cols_order:
            if col not in data_pengguna_agregat.columns:
                data_pengguna_agregat[col] = pd.NA # atau 0, tergantung bagaimana model Anda menangani missing value

        # Pastikan urutan kolom benar
        data_pengguna_agregat = data_pengguna_agregat[final_cols_order]

        # --- Akhir Sisipkan Kode Pemrosesan Data Anda di Sini ---


        # Tombol untuk melakukan Prediksi
        if st.button("Prediksi Waktu Marathon"):
            try:
                # Lakukan prediksi
                prediction = loaded_model.predict(data_pengguna_agregat)
                prediction_seconds = prediction[0] # Hasil prediksi dalam detik

                # Simpan hasil prediksi di session_state
                st.session_state['prediction_seconds'] = prediction_seconds
                st.session_state['show_pacing_strategy'] = False # Reset status tampilan strategi

                # Tampilkan hasil prediksi waktu total dan average pace
                MARATHON_DISTANCE_KM = 42.195 # Konstanta

                predicted_avg_pace_seconds_per_km = prediction_seconds / MARATHON_DISTANCE_KM
                minutes_pace = int(predicted_avg_pace_seconds_per_km // 60)
                seconds_pace = int(predicted_avg_pace_seconds_per_km % 60)
                predicted_avg_pace_str = f"{minutes_pace}:{seconds_pace:02d} per km"

                hours_total = int(prediction_seconds // 3600)
                minutes_total = int((prediction_seconds % 3600) // 60)
                seconds_total = int(prediction_seconds % 60)
                predicted_time_str = f"{hours_total:02d}:{minutes_total:02d}:{seconds_total:02d}"

                st.subheader("Hasil Prediksi Marathon Anda:")
                st.write(f"Waktu penyelesaian prediksi: **{predicted_time_str}**")
                st.write(f"Untuk mencapai waktu ini, Anda perlu menjaga rata-rata pacing sekitar: **{predicted_avg_pace_str}**")


            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
                st.session_state['prediction_seconds'] = None # Reset prediksi jika ada error
                st.session_state['show_pacing_strategy'] = False
    
    except Exception as e:
        # Tangani error apa pun yang terjadi selama pemrosesan data file
        st.error(f"Terjadi kesalahan saat memproses file Anda: {e}")
        st.info("Harap pastikan file CSV Garmin Anda memiliki format dan kolom yang benar.")
        st.session_state['prediction_seconds'] = None # Reset prediksi jika ada error
        st.session_state['show_pacing_strategy'] = False

# --- Bagian Tampilan Strategi Pacing (Ditampilkan hanya jika user mengklik tombol) ---

# Tombol untuk menampilkan Strategi Pacing (Muncul hanya jika prediksi sudah ada)
if st.session_state['prediction_seconds'] is not None:
    if st.button("Tampilkan Strategi Pacing"):
        st.session_state['show_pacing_strategy'] = True # Set status untuk menampilkan strategi


# Logika untuk menampilkan strategi pacing hanya jika show_pacing_strategy True
if st.session_state['show_pacing_strategy']:
    st.subheader("Saran Strategi Pacing:")
    st.write("Atur slider di bawah untuk melihat perkiraan pacing berdasarkan strategi split:")

    # Inisialisasi session_state untuk slider pacing jika belum ada
    if 'split_percentage_pacing' not in st.session_state:
        st.session_state['split_percentage_pacing'] = 0.0

    # Slider untuk mengatur split dengan key unik
    split_percentage = st.slider("Pilih Persentase Split (+ untuk Positif, - untuk Negatif)",
                                 min_value=-5.0, max_value=5.0,
                                 value=st.session_state['split_percentage_pacing'], # Set nilai awal dari session_state
                                 step=0.1,
                                 key='split_pacing_slider_widget') # <<=== Key unik untuk slider

    # Update session_state saat nilai slider berubah (ini otomatis dengan key)
    # st.session_state['split_percentage_pacing'] akan diupdate oleh Streamlit

    # Ambil nilai prediksi dari session_state
    prediction_seconds = st.session_state['prediction_seconds']
    MARATHON_DISTANCE_KM = 42.195

    # Perhitungan pacing berdasarkan nilai slider dari session_state
    current_split_percentage = st.session_state['split_pacing_slider_widget'] # Ambil nilai dari session_state menggunakan key widget

    if (2 + current_split_percentage/100) != 0:
        time_half_1 = prediction_seconds / (2 + current_split_percentage/100)
        time_half_2 = prediction_seconds / (2 - current_split_percentage/100)

        # Pastikan pembagian dengan nol tidak terjadi untuk pace
        if (MARATHON_DISTANCE_KM / 2) != 0:
             pace_half_1_s_km = time_half_1 / (MARATHON_DISTANCE_KM / 2)
             pace_half_2_s_km = time_half_2 / (MARATHON_DISTANCE_KM / 2)

             # Konversi ke format mm:ss
             pace_half_1_str = f"{int(pace_half_1_s_km // 60)}:{int(pace_half_1_s_km % 60):02d} per km"
             pace_half_2_str = f"{int(pace_half_2_s_km // 60)}:{int(pace_half_2_s_km % 60):02d} per km"

             st.write(f"Jika menggunakan strategi split {current_split_percentage:.1f}%:")
             st.write(f"- Paruh pertama (~21.1 km): jaga pacing sekitar **{pace_half_1_str}**")
             st.write(f"- Paruh kedua (~21.1 km): jaga pacing sekitar **{pace_half_2_str}**")
             st.write("*(Perhitungan ini adalah perkiraan sederhana dan tidak memperhitungkan faktor-faktor lain secara rinci)*")
        else:
            st.warning("Tidak dapat menghitung pacing karena jarak marathon tidak valid.")
    else:
         st.warning("Strategi split tidak valid.")