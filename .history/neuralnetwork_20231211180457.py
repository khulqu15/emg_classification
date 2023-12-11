import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix
import seaborn as sns
print(tf.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

file_path = './data.xlsx'

try:
    # Mengambil dan menampilkan data
    data = pd.read_excel(file_path)
    selected_columns = data.columns[:3]
    data_selected = data[selected_columns]
    print("Dataset berupa: --------------------------- \n")
    print(data_selected.head())
    
    # Pemisahan data klasifikasi : Fleksi (1-50) dan Ekstensi (51-100)
    fleksi_data = data_selected.iloc[0:50].copy()
    ekstensi_data = data_selected.iloc[50:100].copy()
    
    # Membuat label untuk Fleksi: 0 dan Ekstensi: 1
    fleksi_data['Label'] = 0
    ekstensi_data['Label'] = 1
    combined_data = pd.concat([fleksi_data, ekstensi_data]) # Menggabungkan dataset
    print("\nPemisahan data menjadi: --------------------------- \n")
    print(combined_data.head())
    
    # Pra-pemrosesan data: Konversi Fleksi dan Ekstensi ke numerik
    combined_data['Fleksi'] = combined_data['Fleksi'].str.replace(',', '.').astype(float)
    combined_data['Ekstensi'] = combined_data['Ekstensi'].str.replace(',', '.').astype(float)    
    
    # Pemisahan fitur (X) dan label (y)
    combined_data = combined_data.drop('Time', axis=1)
    X = combined_data.drop('Label', axis=1)
    y = combined_data['Label']
    # Membagi data menjadi training dan validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalisasi data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nHasil Pra-Pemrosesan data: --------------------------- \n")
    print(X_train_scaled[:5])
    print(y_train.head())
    
    def apply_low_pass_filter(signal, cutoff=60, fs=1000, order=6):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    # Menghitung nilai absolut dari sinyal EMG
    abs_signal = np.abs(combined_data[['Fleksi', 'Ekstensi']])

    # Menerapkan Linear Envelope (filter low-pass)
    linear_envelope_signal = abs_signal.apply(lambda x: apply_low_pass_filter(x))

    # Plotting sinyal asli dan Linear Envelope
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(combined_data['Fleksi'], label='Original Signal')
    plt.title('Original EMG Signal')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(linear_envelope_signal['Fleksi'], label='Linear Envelope', color='orange')
    plt.title('EMG Signal After Linear Envelope')
    plt.legend()
    print(combined_data['Fleksi'])
    print(linear_envelope_signal['Fleksi'])
    plt.tight_layout()
    plt.show()
    
    # Neural Network
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    # Nueral Network: Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # Neural Network: Train with training data
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=20, batch_size=10)
    # Neural Network: Evaluate validation data
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print("\nHasil Neural Network: --------------------------- \n")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Plotting: Get data history
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)
    # Plotting: Plotting process
    plt.figure(figsize=(12, 16))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Membuat prediksi pada data pengujian
    y_pred_nn = model.predict(X_test_scaled)
    y_pred_nn = [1 if prob > 0.5 else 0 for prob in y_pred_nn]
    # Menghasilkan matriks konfusi
    conf_mat_nn = confusion_matrix(y_test, y_pred_nn)
    # Plotting matriks konfusi
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat_nn, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Neural Network')
    plt.show()
    
except Exception as e:
    print(e)