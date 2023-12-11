import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

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
    
    with tf.device('/cpu:0'):
        # Neural Network
        model = Sequential([
            Dense(10, activation='relu', input_shape=(X_train_scaled[1],)),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid')
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
except Exception as e:
    print(e)