import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Faylı oxu
file_path = "Raman 2001_Quadrupole deformation data_final.xlsx"
df = pd.read_excel(file_path, sheet_name="Sayfa4")
# Sütun adlarını təmizləyək (əgər lazım olarsa)
df.columns = df.columns.str.strip()

# Sayısal sütunlardakı dəyərləri təmizləyək
def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace(",", ".")  # Vergülü nöqtəyə çevir
        x = x.replace(" ", "")   # Boşluqları sil
    try:
        return float(x)
    except ValueError:
        return np.nan  # Əgər çevirmək mümkün deyilsə, NaN təyin et

# İlgili sütunları seçək
input_features = ["A", "Z", "N"]
target_columns = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# Sütunları təmizləyək
for col in input_features + target_columns:
    df[col] = df[col].apply(clean_numeric)


# Boş olmayan verilənləri seçək
df_filtered = df.dropna(subset=input_features)


# Boş olan sütunları doldurmaq üçün dövr
for target in target_columns:
    df_train = df_filtered.dropna(subset=[target])  # Yalnız hədəf sütunu dolu olanları götür
    df_test = df_filtered[df_filtered[target].isna()]  # Hədəf sütunu boş olanları götür
    
    if df_test.empty:
        continue  # Əgər boş xanalar yoxdursa, keç
    
    X_train = df_train[input_features]
    y_train = df_train[target]
    
    X_test = df_test[input_features]

    # Model qur
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Boş dəyərləri proqnozlaşdır
    predicted_values = model.predict(X_test)

    # Proqnozları dataframedə əvəz et
    df.loc[df[target].isna(), target] = predicted_values

# Yenilənmiş dataset-i saxla
df.to_excel("Updated_Dataset.xlsx", index=False)

print("Boş xanalar Random Forest modeli ilə dolduruldu. Yeni fayl: 'Updated_Dataset.xlsx'")






