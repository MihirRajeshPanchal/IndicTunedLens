from datasets import load_dataset
import pandas as pd

language = "te"
ds = load_dataset("alexandrainst/m_mmlu", language, split="train")

records = []

for row in ds:
    question = row["instruction"]

    # create list of options
    options = [
        row["option_a"],
        row["option_b"],
        row["option_c"],
        row["option_d"]
    ]


    answer_letter = row["answer"].strip().upper()
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}

    correct_idx = mapping[answer_letter]
    answer_text = options[correct_idx]

    answer_key = answer_letter.lower()

    records.append({
        "instruction": question,
        "options": options,
        "answer": answer_text,
        "answer_key": answer_key
    })

df = pd.DataFrame(records)

df.to_csv(f"/mnt/storage/deeksha/indictunedlens/data/m_mmlu_{language}.csv", index=False, encoding="utf-8")

print("Saved", len(df), f"rows to m_mmlu_{language}.csv")