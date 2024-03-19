import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

data_path = './data.xlsx'
data = pd.read_excel(data_path)
# print(data.head())

sheet_names = pd.ExcelFile(data_path).sheet_names
# print(sheet_names)

all_data = pd.DataFrame()
for sheet in sheet_names:
    sheet_data = pd.read_excel(data_path, sheet_name=sheet, usecols=['Time', 'Fleksi', 'Ekstensi'])
    all_data = pd.concat([all_data, sheet_data], ignore_index=True)
print('Data (shape): ', all_data.shape)

all_data['Fleksi'] = all_data['Fleksi'].str.replace(',', '.').astype(float)
all_data['Ekstensi'] = all_data['Ekstensi'].str.replace(',', '.').astype(float)
scaler = StandardScaler()
all_data = all_data.dropna(subset=['Fleksi', 'Ekstensi'])
features = scaler.fit_transform(all_data[['Fleksi', 'Ekstensi']])
kmeans = KMeans(n_clusters=2, random_state=42)
all_data['Cluster'] = kmeans.fit_predict(features)
cluster_centers = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers)
# print(cluster_centers)
# Kluster 1: Fleksi ≈ 70.71, Ekstensi ≈ 43.24
# Kluster 2: Fleksi ≈ 267.42, Ekstensi ≈ 185.94

labels = all_data['Cluster']
X = all_data[['Fleksi', 'Ekstensi']]
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train data (dataset): ', X_train.shape, y_train.shape)
print('Test data (validation): ', X_test.shape, y_test.shape)


## KMeans
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=0)
final_accuracy = history.history['val_accuracy'][-1]
model.evaluate(X_test, y_test, verbose=1)
print(final_accuracy)

predictions = model.predict(X_test)
predictions = np.where(predictions > 0.5, 1, 0)
cm = confusion_matrix(y_test, predictions)
print(cm)
labels = ['Fleksi', 'Ekstensi']
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r', square=True, linewidths=.5, xticklabels=labels, yticklabels=labels)
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.title('Matriks Konfusi', size=15)
plt.show()


## CNN
model_cnn = keras.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
_, accuracy = model_cnn.evaluate(X_test, y_test, verbose=1)
print(f'Accuracy: {accuracy}')

predictions = model_cnn.predict(X_test)
predictions = np.where(predictions > 0.5, 1, 0)
cm = confusion_matrix(y_test, predictions)

labels = ['Fleksi', 'Ekstensi']
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r', square=True, linewidths=.5, xticklabels=labels, yticklabels=labels)
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.title('Matriks Konfusi', size=15)
plt.show()


## KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
print('Accuracy: ', accuracy)
print('Error rate: ', error_rate)

cm = confusion_matrix(y_test, y_pred)
labels = ['Fleksi', 'Ekstensi']
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r', square=True, linewidths=.5, xticklabels=labels, yticklabels=labels)
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.title('Matriks Konfusi', size=15)
plt.show()