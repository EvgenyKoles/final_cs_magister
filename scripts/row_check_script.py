import pandas as pd
from pathlib import Path

# Путь к data/processed относительно текущего скрипта
p = Path(__file__).resolve().parents[1] / "data" / "processed"

splits = ["train", "val", "test"]

for split in splits:
    X_path, y_path = p/f"X_{split}.csv", p/f"y_{split}.csv"

    print(f"\n=== {split.upper()} ===")
    print(f"X_path: {X_path.exists()}, y_path: {y_path.exists()}")

    if not X_path.exists() or not y_path.exists():
        print("⚠️ Файл(ы) не найдены, пропускаем")
        continue

    # читаем как есть
    X_raw = pd.read_csv(X_path, header=None)
    y_raw = pd.read_csv(y_path, header=None)

    print(f"X_raw shape: {X_raw.shape}, y_raw shape: {y_raw.shape}")
    print("X first row:", list(X_raw.iloc[0, :5]))
    print("y first rows:\n", y_raw.head(3))

    # убираем заголовок в y
    if y_raw.iloc[0,0] == "label":
        y_data = y_raw.iloc[1:].astype(int)
    else:
        y_data = y_raw.astype(int)

    print(f"After cleaning -> X={X_raw.shape[0]}, y={y_data.shape[0]}")
    if X_raw.shape[0] != y_data.shape[0]:
        print("⚠️ Несовпадение! Нужно skiprows=1 в X или y.")
    else:
        print("✅ X и y синхронизированы")
