import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

# Dataseti yükləyirik
file_path = "Raman 2001_Quadrupole deformation data_final.xlsx"  # Fayl adını uyğun dəyişin
df = pd.read_excel(file_path, sheet_name=0)  # Əgər səhifə adı fərqlidirsə, onu dəyişin

# Lazımi sütunları seçirik
features = ["A", "Z", "N"]
targets = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# Sütunlardakı dəyərləri rəqəmsal formata çevirmək
for col in targets:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Rəqəmə çevirmək

# Hər sütunun proqnozlaşdırılmış versiyasını əlavə edirik
for target in targets:
    df_train = df[features + [target]].dropna()
    df_test = df[features]  # Bütün verilənlər üçün proqnoz veririk

    if not df_train.empty:
        X_train, X_valid, y_train, y_valid = train_test_split(
            df_train[features], df_train[target], test_size=0.2, random_state=42
        )

        # Modeli qururuq
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Bütün verilənlər üçün proqnozlar əldə edirik
        predicted_values = model.predict(df_test)
        df[f"{target}_predicted"] = predicted_values

# Yenilənmiş dataset'i saxlayırıq
df.to_excel("Updated_Dataset.xlsx", index=False)
print("Proqnozlar tamamlandı! Yenilənmiş dataset 'Updated_Dataset.xlsx' olaraq saxlanıldı.")
