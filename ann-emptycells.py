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

# Modelin nəticələrini saxlamaq üçün DataFrame yaradılır
predicted_values = df.copy()

# Hər sütun üçün modeli öyrədirik və boş xanaları proqnozlaşdırırıq
for feature in output_features:
    # Boş olmayan sətirləri seçirik
    train_data = df.dropna(subset=[feature])
    
    # Giriş və çıxış dəyərlərini ayırırıq
    X_train = train_data[input_features]
    y_train = train_data[feature]
    
    # MLPRegressor modelini qururuq
    model = Pipeline([
        ("scaler", StandardScaler()), 
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42))
    ])
    
    # Modeli öyrədirik
    model.fit(X_train, y_train)
    
    # Boş xanaları olan sətirləri seçirik
    missing_data = df[df[feature].isna()]
    
    # Əgər boş xanalar yoxdursa, keçirik
    if missing_data.empty:
        continue
    
    # Giriş dəyərlərini seçirik
    X_missing = missing_data[input_features]
    
    # Proqnozları əldə edirik
    predicted_values.loc[df[feature].isna(), feature] = model.predict(X_missing)

# Proqnozlaşdırılmış dataset-i saxlayırıq
predicted_values.to_excel("Predicted_Data.xlsx", index=False)

print("Proqnozlaşdırılmış nəticələr 'Predicted_Data.xlsx' faylına yazıldı.")
