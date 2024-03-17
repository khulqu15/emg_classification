import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

file_path = './data.xlsx'
xls = pd.ExcelFile(file_path)
all_data = pd.DataFrame()
columns = ['Fleksi', 'Ekstensi']

for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    df = df[['Fleksi', 'Ekstensi']]
    all_data = pd.concat([all_data, df], ignore_index=True)

for sheet, data in all_data.items():
    print(f"Sheet: {sheet}")
    print(data)
    print("\n")

all_data['Fleksi'] = all_data['Fleksi'].str.replace(',', '.').astype(float)
all_data['Ekstensi'] = all_data['Ekstensi'].str.replace(',', '.').astype(float)

X = all_data['Ekstensi'].values.reshape(-1, 1)
y = np.where(all_data['Fleksi'] > all_data['Ekstensi'], 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

model = keras.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train_reshaped.shape[1], 1), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=1),
    keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=1),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid'),
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test_reshaped, y_test), callbacks=[early_stopping])
model.evaluate(X_test_reshaped, y_test, verbose=1)