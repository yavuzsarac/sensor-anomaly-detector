import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
def system(file_path):
    # ------------------------------------------------------------------------------
    # Seed
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # ------------------------------------------------------------------------------
    # Veriyi yükleme ve bölme
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path, header=None).T
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON list tipindeyse direk DataFrame'e çevir
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # İlk anahtarın içindeki listeyi al
            first_key = list(data.keys())[0]
            df = pd.DataFrame(data[first_key])
        else:
            raise ValueError("JSON formatı desteklenmiyor.")
    else:
        raise ValueError("Sadece CSV veya JSON dosyaları desteklenir.")

    df.columns = ['sensor_value']
    df.columns = ['sensor_value']
    values = df['sensor_value'].values.reshape(-1, 1)

    # Toplam 5000 veri: son 400 test, kalan 4600 eğitim
    train_val_data = values[:4600]
    test_data = values[4600:]

    # ------------------------------------------------------------------------------
    # Normalizasyon 
    normalizer = Normalization()
    normalizer.adapt(train_val_data)

    # ------------------------------------------------------------------------------
    # Autoencoder 
    input_layer = Input(shape=(1,))
    x = normalizer(input_layer)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    bottleneck = Dense(4, activation='relu')(x)
    x = Dense(8, activation='relu')(bottleneck)
    x = Dense(16, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # ------------------------------------------------------------------------------
    # Model eğitimi
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = autoencoder.fit(train_val_data, train_val_data, validation_split=0.2, epochs=150, batch_size=32, callbacks=[early_stop],  verbose=1)

    # ------------------------------------------------------------------------------
    # Ağırlıkların kaydedilmesi
    autoencoder.save_weights("anomaly_weights.weights.h5")
    print("weights_saved ")

    # ------------------------------------------------------------------------------
    # Eşik belirlemek için train_val verisi üzerinde tahmin
    reconstructed_train_val = autoencoder.predict(train_val_data)
    mse_train_val = np.mean(np.power(train_val_data - reconstructed_train_val, 2), axis=1)

    # Anomali eşik değeri 
    threshold = np.mean(mse_train_val) + 20 * np.std(mse_train_val)

    # ------------------------------------------------------------------------------
    # Test verisi üzerinde tahmin ve MSE
    reconstructed_test = autoencoder.predict(test_data)
    mse_test = np.mean(np.power(test_data - reconstructed_test, 2), axis=1)

    # Anomali tespiti
    anomalies_test = mse_test > threshold

    # ------------------------------------------------------------------------------
    # Performans metrikleri 
    mse_score = mean_squared_error(test_data, reconstructed_test)
    mae_score = mean_absolute_error(test_data, reconstructed_test)

    # ------------------------------------------------------------------------------
    # Grafik 1: Loss vs Validation Loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------
    # Grafik 2: MSE Histogramı ve Eşik Değeri
    plt.figure(figsize=(10, 4))
    plt.hist(mse_train_val, bins=100, color='skyblue')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.6f}')
    plt.title("Train+Val MSE Dağılımı ve Anomali Eşiği")
    plt.xlabel("MSE")
    plt.ylabel("Frekans")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------
    # Grafik 3: Test Verisi Üzerinde Anomali Noktaları
    plt.figure(figsize=(14, 5))
    plt.plot(test_data, label='Test Verisi', linewidth=1)
    plt.scatter(np.arange(len(test_data))[anomalies_test], test_data[anomalies_test], color='red', label='Anomali', s=30)
    plt.title("Test Verisi Üzerinde Anomali Tespiti")
    plt.xlabel("Zaman")
    plt.ylabel("Sensör Değeri")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------
    # Grafik 4: MSE ve MAE Çubuk Grafiği
    plt.figure(figsize=(6, 4))
    plt.bar(['MSE', 'MAE'], [mse_score, mae_score], color=['orange', 'green'])
    plt.title('Test Verisi Üzerinde Rekonstrüksiyon Hataları')
    plt.ylabel('Hata Değeri')
    for i, v in enumerate([mse_score, mae_score]):
        plt.text(i, v + v*0.02, f"{v:.6f}", ha='center', fontweight='bold')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------
    # Veriseti istatistikleri:

    dataset_median = df['sensor_value'].median()
    dataset_std = df['sensor_value'].std()
    min_val = df['sensor_value'].min()
    max_val = df['sensor_value'].max()
    dataset_mean = df['sensor_value'].mean()
    dataset_mode = df['sensor_value'].mode()[0]

    # ------------------------------------------------------------------------------
    # Sonuç
    print("VERİLER İLE İLGİLİ DEĞERLER:")
    print(f"Test verisinde toplam {np.sum(anomalies_test)} adet anomali tespit edildi.\n")
    print(f"MAE: {mae_score:.6f}\n")
    print(f"MSE: {mse_score:.6f}\n" )
    print(f"Eşik değer (threshold): {threshold:.6f}\n")
    print(f"Verilerin medyanı: {dataset_median}\n")
    print(f"Verilerin standart sapması: {dataset_std}\n")
    print(f"Minimum değer: {min_val}\n")
    print(f"Maksimum değer: {max_val}\n")
    print(f"Verilerin ortalaması: {dataset_mean}\n")
    print(f"En çok tekrar eden değer: {dataset_mode}\n")