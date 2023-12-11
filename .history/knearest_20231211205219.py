import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
import seaborn as sns
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os
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
    
    def apply_low_pass_filter(signal, cutoff=2, fs=20, order=2):
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

    # Untuk Fleksi
    plt.plot(combined_data['Fleksi'], label='Original Signal - Fleksi')
    plt.plot(linear_envelope_signal['Fleksi'], label='Linear Envelope - Fleksi', linestyle='dashed')

    # Untuk Ekstensi
    plt.plot(combined_data['Ekstensi'], label='Original Signal - Ekstensi')
    plt.plot(linear_envelope_signal['Ekstensi'], label='Linear Envelope - Ekstensi', linestyle='dashed')

    plt.title('EMG Signal Comparison')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    param_grid = {'n_neighbors': range(1, 11)} # Mengatur grid param pengujian
    knn_cv = GridSearchCV(knn, param_grid, cv=5) # Mencari parameter terbaik
    knn_cv.fit(X_train_scaled, y_train) # Melatih model
    best_params = knn_cv.best_params_
    best_score = knn_cv.best_score_
    print("\nHasil KNN: --------------------------- \n")
    print("Best score: " + str(best_params))
    print("Best score: " + str(best_score))
    # KNN evaluate model
    knn_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
    knn_best.fit(X_train_scaled, y_train)
    y_pred_best = knn_best.predict(X_test_scaled)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    print("Best Accuracy: " + str(accuracy_best))
    
    # Plot Nilai Akurasi vs Jumlah Tetangga
    neighbors = range(1, 11)
    cv_scores = knn_cv.cv_results_['mean_test_score']

    plt.figure(figsize=(10, 5))
    plt.plot(neighbors, cv_scores, marker='o')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('CV Accuracy')
    plt.title('KNN Accuracy vs Number of Neighbors')
    plt.xticks(neighbors)
    plt.grid(True)
    plt.show()

    # Matriks Konfusi untuk model terbaik
    conf_mat = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for KNN')
    plt.show()
    
    # Pra-pemrosesan Data untuk KNN
    X_train_le, X_test_le, y_train_le, y_test_le = train_test_split(linear_envelope_signal, y, test_size=0.2, random_state=42)

    # Normalisasi data
    scaler_le = StandardScaler()
    X_train_le_scaled = scaler_le.fit_transform(X_train_le)
    X_test_le_scaled = scaler_le.transform(X_test_le)

    # Pelatihan dan Optimasi Model KNN
    knn_le = KNeighborsClassifier()
    knn_cv_le = GridSearchCV(knn_le, param_grid, cv=5)
    knn_cv_le.fit(X_train_le_scaled, y_train_le)

    best_params_le = knn_cv_le.best_params_
    best_score_le = knn_cv_le.best_score_

    # Evaluasi Model
    knn_best_le = KNeighborsClassifier(n_neighbors=best_params_le['n_neighbors'])
    knn_best_le.fit(X_train_le_scaled, y_train_le)
    y_pred_best_le = knn_best_le.predict(X_test_le_scaled)
    accuracy_best_le = accuracy_score(y_test_le, y_pred_best_le)

    # Output dan Plotting
    print("\nHasil KNN dengan Linear Envelope: --------------------------- \n")
    print("Best parameters:", best_params_le)
    print("Best cross-validation score:", best_score_le)
    print("Best test accuracy:", accuracy_best_le)

    # Plotting Nilai Akurasi vs Jumlah Tetangga
    neighbors_le = range(1, 11)
    cv_scores_le = knn_cv_le.cv_results_['mean_test_score']

    plt.figure(figsize=(10, 5))
    plt.plot(neighbors_le, cv_scores_le, marker='o')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('CV Accuracy')
    plt.title('KNN with Linear Envelope: Accuracy vs Number of Neighbors')
    plt.xticks(neighbors_le)
    plt.grid(True)
    plt.show()

    # Matriks Konfusi
    conf_mat_le = confusion_matrix(y_test_le, y_pred_best_le)

    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_mat_le, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for KNN with Linear Envelope')
    plt.show()
    
    print("\nEvaluasi mendalam model KNN: --------------------------- \n")
    print("KNN dengan Data Asli:")
    print(classification_report(y_test, y_pred_best))
    print("KNN dengan Linear Envelope:")
    print(classification_report(y_test_le, y_pred_best_le))

except Exception as e:
    print(e)