from sklearn.model_selection import train_test_split

# 3 ayrı qrafik çəkək (hər bir dəyişən üçün) - yalnız nöqtələrlə
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, target in enumerate(target_columns):
    ax = axes[i]
    ax.scatter(df_sample.index, df_sample[target], label="Actual", color='blue', marker="o", alpha=0.7)
    ax.scatter(df_sample.index, df.loc[df_sample.index, target], label="Predicted", color='orange', marker="x", alpha=0.7)

    ax.set_title(f"{target} - Actual vs Predicted")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


