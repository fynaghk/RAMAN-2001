import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# Faylın oxunması
file_path = "Raman 2001_Quadrupole deformation data_final.xlsx"
df = pd.read_excel(file_path)

# Dəyərləri düzəltmək üçün vergülləri nöqtə ilə əvəz edirik və string-ləri float-a çeviririk
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "")
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Giriş və çıxış sütunları
input_features = ["A", "Z", "N"]
output_features = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# Modelin nəticələrini saxlamaq üçün yeni sütunlar yaradırıq
for feature in output_features:
    df[f"Predicted_{feature}"] = np.nan  # Yeni sütunlar

# Modeli hər sütun üçün öyrədib bütün dəyərləri proqnozlaşdırırıq
for feature in output_features:
    # Giriş və çıxış dəyərlərini ayırırıq
    X = df[input_features]
    y = df[feature]
    
    # Məlumatları bölürük (boş olanlar üçün də proqnoz verəcəyik)
    X_train = X[y.notna()]
    y_train = y[y.notna()]
    X_predict = X  # Bütün dataset üçün proqnoz verəcəyik
    
    # MLPRegressor modelini qururuq
    model = Pipeline([
        ("scaler", StandardScaler()), 
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42))
    ])
    
    # Modeli öyrədirik
    model.fit(X_train, y_train)
    
    # Bütün sətirlər üçün proqnoz veririk
    df[f"Predicted_{feature}"] = model.predict(X_predict)

# Proqnozlaşdırılmış dataset-i saxlayırıq
df.to_excel("Predicted_Data2.xlsx", index=False)

print("Proqnozlaşdırılmış nəticələr 'Predicted_Data.xlsx' faylına yazıldı.")
