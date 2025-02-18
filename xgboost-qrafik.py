import matplotlib.pyplot as plt

# Random nümunə seçirik ki, qrafik daha aydın görünsün
df_sample = df.sample(n=100, random_state=42)  # 100 sətir seçirik

# Qrafik çəkmək üçün sütun adlarını müəyyən edirik
target_columns = ["β2", "Q0(b)", "B(E2) (e2b2)"]

# 3 ayrı qrafik çəkirik
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(target_columns):
    ax = axes[i]
    ax.scatter(df_sample.index, df_sample[target], label="Actual", color='blue', marker="o", alpha=0.7)
    ax.scatter(df_sample.index, df.loc[df_sample.index, f"{target}_predicted"], label="Predicted", color='orange', marker="x", alpha=0.7)

    ax.set_title(f"{target} - Actual vs Predicted")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
