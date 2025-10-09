# scan_all_cicids.py
# Scan a folder of CIC-IDS2017 CSVs, print label counts per file and combined,
# show a combined bar chart (and optional per-file plots).

import os
import sys
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt


data_dir = "/Users/igor/Downloads/MachineLearningCVE" 
plot_per_file = False 
show_combined_plot = True


if not os.path.isdir(data_dir):
    print("ERROR: data_dir does not exist:", data_dir)
    sys.exit(1)

# find csv files
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.lower().endswith(".csv")]
if not csv_files:
    print("No CSV files found in", data_dir)
    sys.exit(0)

print(f"Found {len(csv_files)} CSV files in {data_dir}\n")

combined_counts = Counter()
per_file_counts = {}

def find_label_column(columns):
    # try direct match 'label' ignoring case and surrounding spaces
    for c in columns:
        if c.strip().lower() == "label":
            return c
    # fallback: any column name containing 'label' (case-insensitive)
    for c in columns:
        if "label" in c.lower():
            return c
    return None

for fp in sorted(csv_files):
    fname = os.path.basename(fp)
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as e:
        print(f"Failed to read {fname}: {e}")
        continue

    # normalize column names (strip only, but keep originals for detection)
    df.columns = [c.strip() for c in df.columns]

    label_col = find_label_column(df.columns)
    if label_col is None:
        print(f"{fname}: no label column found (skipping file).")
        continue

    # clean labels: strip whitespace, unify case
    labels = df[label_col].astype(str).str.strip()
    # normalize common variants
    labels = labels.replace({
        "BENIGN": "Benign",
        "BENIGN ": "Benign",
        "benign": "Benign",
        "DDoS": "DDoS",
        "ddos": "DDoS",
        "DOS": "DoS",
        "dos": "DoS"
    })

    counts = labels.value_counts(dropna=True)
    per_file_counts[fname] = counts.to_dict()

    # print per-file counts
    print(f"File: {fname}")
    print(counts.to_string())
    print("-" * 40)

    # update combined
    for k, v in counts.items():
        combined_counts[k] += int(v)

    # optional per-file plot
    if plot_per_file:
        plt.figure(figsize=(8,4))
        counts.sort_values(ascending=False).plot(kind="bar")
        plt.title(f"Label counts — {fname}")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

# Combined summary
if combined_counts:
    print("\nCOMBINED COUNTS ACROSS ALL FILES")
    combined_series = pd.Series(dict(combined_counts)).sort_values(ascending=False)
    print(combined_series.to_string())

    if show_combined_plot:
        plt.figure(figsize=(10,5))
        bars = plt.bar(combined_series.index.tolist(), combined_series.values)
        plt.title("Combined label counts across all CSVs")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        # annotate
        maxv = combined_series.values.max()
        for bar, val in zip(bars, combined_series.values):
            plt.text(bar.get_x() + bar.get_width()/2, val + maxv*0.005, f"{int(val):,}",
                     ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()
else:
    print("No labels collected from files.")
