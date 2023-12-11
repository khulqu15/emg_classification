import pandas as pd

file_path = './data.xlsx'

try:
    # Mengambil dan menampilkan data
    data = pd.read_excel(file_path)
    selected_columns = data.columns[:3]
    data_selected = data[selected_columns]
    print(data_selected)
    
    # Pemisahan data klasifikasi : Fleksi (1-50) dan Ekstensi (51-100)
    fleksi_data = data_selected.iloc[0:50]
    ekstensi_data = data_selected.iloc[50:100]
    
    # Membuat label untuk Fleksi: 0 dan Ekstensi: 1
    fleksi_data['Label'] = 0
    ekstensi_data['Label'] = 1
    combined_data = pd.concat([fleksi_data, ekstensi_data]) # Menggabungkan dataset
    
except Exception as e:
    print(e)