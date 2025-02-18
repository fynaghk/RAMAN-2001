import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 🔹 Faylı oxu
file_path = "Raman 2001_Quadrupole deformation data_final.xlsx"
df = pd.read_excel(file_path, sheet_name="Sayfa4")

# 🔹 Sütun adlarını təmizləyək (əgər boşluqlar varsa)
df.columns = df.columns.str.strip()

# 🔹 Sütunlardakı rəqəm formatını düzəltmək funksiyası
def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace(",", ".")  # Vergülü nöqtəyə çevir
        x = x.replace(" ", "")   # Boşluqları sil
    try:
        return float(x)
    except ValueError:
        return np.nan  # Əgər çevirmək mümkün deyilsə, NaN təyin et

# 🔹 İstifadə olunan sütunlar
input_features = ["A", "Z", "N"]
target_columns = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# 🔹 Bütün seçilmiş sütunları təmizləyək
for col in input_features + target_columns:
    df[col] = df[col].apply(clean_numeric)

# 🔹 Bütün dataset üzrə proqnozlaşdırma üçün model
for target in target_columns:
    df_train = df.dropna(subset=target_columns)  # Həm giriş, həm də hədəf dəyərləri dolu olanları götür
    
    if df_train.empty:
        continue  # Əgər dolu məlumat yoxdursa, keç
    
    X_train = df_train[input_features]
    y_train = df_train[target]

    # 🔹 Model qur
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 🔹 Bütün dəyərləri proqnozlaşdır
    df[target] = model.predict(df[input_features])

# 🔹 Yenilənmiş dataset-i saxla
df.to_excel("Predicted_Dataset.xlsx", index=False)

print("Bütün 'β2', 'Q0(b)', 'B(E2) (e2b2)' dəyərləri Random Forest ilə proqnozlaşdırıldı. Yeni fayl: 'Predicted_Dataset.xlsx'")
