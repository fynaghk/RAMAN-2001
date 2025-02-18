import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Faylın oxunması
file_path = "Raman 2001_Quadrupole deformation data_final.xlsx"
df = pd.read_excel(file_path)

# Dəyərləri düzəltmək üçün vergülləri nöqtə ilə əvəz edirik və string-ləri float-a çeviririk
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "")
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Giriş və çıxış sütunları
input_features = ["A", "Z", "N"]
target_columns = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# Modelin nəticələrini saxlamaq üçün yeni sütunlar yaradırıq
for feature in target_columns:
    df[f"Predicted_{feature}"] = np.nan  # Yeni sütunlar

# Modeli hər sütun üçün öyrədib bütün dəyərləri proqnozlaşdırırıq
performance_metrics = {}

for feature in target_columns:
    # Giriş və çıxış dəyərlərini ayırırıq
    X = df[input_features]
    y = df[feature]
    
    # Boş olmayan dəyərləri ayırırıq
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
    y_pred = model.predict(X_predict)
    df[f"Predicted_{feature}"] = y_pred
    
    # Modelin performansını hesablayırıq
    y_actual = y[y.notna()]
    y_pred_actual = model.predict(X_train)  # Yalnız faktiki dəyərlər üçün proqnoz
    
    mae = mean_absolute_error(y_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
    r2 = r2_score(y_actual, y_pred_actual)
    
    performance_metrics[feature] = {"MAE": mae, "RMSE": rmse, "R²": r2}

# Modelin performansını göstəririk
performance_df = pd.DataFrame(performance_metrics).T
print("Model Performance Metrics:")
print(performance_df)

# Proqnozlaşdırılmış dataset-i saxlayırıq
df.to_excel("Predicted_Data.xlsx", index=False)

# Nümunəvi məlumatların seçilməsi (qrafiklər üçün)
df_sample = df.sample(n=100, random_state=42)  # 100 təsadüfi nümunə götürək

# 3 ayrı qrafik çəkirik (hər bir dəyişən üçün) - yalnız nöqtələrlə
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(target_columns):
    ax = axes[i]
    ax.scatter(df_sample.index, df_sample[target], label="Actual", color='blue', marker="o", alpha=0.7)
    ax.scatter(df_sample.index, df.loc[df_sample.index, f"Predicted_{target}"], label="Predicted", color='orange', marker="x", alpha=0.7)

    ax.set_title(f"{target} - Actual vs Predicted")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Modelin performansını Excel faylına da yazırıq
performance_df.to_excel("Model_Performance.xlsx")

print("Proqnozlaşdırılmış nəticələr 'Predicted_Data.xlsx' və model performansı 'Model_Performance.xlsx' fayllarına yazıldı.")
