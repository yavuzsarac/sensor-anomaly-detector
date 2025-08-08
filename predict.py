import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Normalization

def system(file_path):

    # ------------------------------------------------------------
    # SEED
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # ------------------------------------------------------------
    # VERİYİ YÜKLE (CSV / JSON)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path, header=None).T
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            first_key = list(data.keys())[0]
            df = pd.DataFrame(data[first_key])
        else:
            raise ValueError("JSON formatı desteklenmiyor.")
    else:
        raise ValueError("Sadece CSV veya JSON dosyaları desteklenir.")

    df.columns = ['sensor_value']
    values = df['sensor_value'].values.reshape(-1, 1)

    # ------------------------------------------------------------
    # SPLIT: toplam 5000 varsayımıyla son 400 test
    train_val_data = values[:4600]
    test_data = values[4600:]

    # ------------------------------------------------------------
    # MİMARİ (EĞİTİM YOK)
    input_layer = Input(shape=(1,))
    x = Normalization(name="norm")(input_layer)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    bottleneck = Dense(4, activation='relu')(x)
    x = Dense(8, activation='relu')(bottleneck)
    x = Dense(16, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    # AĞIRLIKLARI YÜKLE
    # (build ile değişkenleri oluşturup sonra load ediyoruz)
    model.build(input_shape=(None, 1))
    model.load_weights("anomaly_weights.weights.h5")
    print("Ağırlıklar yüklendi: anomaly_weights.h5")

    # ------------------------------------------------------------
    # EŞİK (threshold): train+val üzerinde MSE dağılımından
    reconstructed_train_val = model.predict(train_val_data, verbose=0)
    mse_train_val = np.mean(np.power(train_val_data - reconstructed_train_val, 2), axis=1)
    threshold = np.mean(mse_train_val) + 20 * np.std(mse_train_val)
    print(f"Eşik değer (threshold): {threshold:.6f}")

    # ------------------------------------------------------------
    # TEST ÜZERİNDE ANOMALİ TESPİTİ
    reconstructed_test = model.predict(test_data, verbose=0)
    mse_test = np.mean(np.power(test_data - reconstructed_test, 2), axis=1)
    anomalies_test = mse_test > threshold

    # ------------------------------------------------------------
    # GRAFİK: TEST VERİSİ + ANOMALİ NOKTALARI
    plt.figure(figsize=(14, 5))
    plt.plot(test_data, label='Test Verisi', linewidth=1)
    plt.scatter(np.arange(len(test_data))[anomalies_test],
                test_data[anomalies_test], color='red', label='Anomali', s=30)
    plt.title("Test Verisi Üzerinde Anomali Tespiti (Kayıtlı Ağırlıklarla)")
    plt.xlabel("Zaman")
    plt.ylabel("Sensör Değeri")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Test verisinde toplam {np.sum(anomalies_test)} adet anomali tespit edildi.")
