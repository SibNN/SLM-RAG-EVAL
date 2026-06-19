"""Module for excluding all datasets, except one mentioned."""

import pandas as pd

# Параметры
input_path = "data/datasets/common/questions/final_classified.jsonl"
output_path = "data/datasets/single/questions/final_classified.jsonl"
expected_dataset_name = "TAPE_MultiQ"


df = pd.read_json(input_path, lines=True)

# Фильтруем строки по meta["source_dataset"]
df_filtered = df[
    df["meta"].apply(
        lambda m: isinstance(m, dict) and m.get("source_dataset") == expected_dataset_name,
    )
]

# Сохраняем результат
df_filtered.to_json(output_path, orient="records", lines=True, force_ascii=False)

print(f"Сохранено строк: {len(df_filtered)}")
