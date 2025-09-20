import pandas as pd
import numpy as np

# FILE PATHS
FILEPATH = "conn.log.labeled"
OUTPUT_CSV = "conn_preprocessed_onehot.csv"

# Finding the header line with "#fields"
with open(FILEPATH) as f:
    for i, line in enumerate(f):
        if line.startswith("#fields"):
            header_line = i
            columns = line.strip().split("\t")[1:]  # skip '#fields'
            break

# Reading the data
df = pd.read_csv(
    FILEPATH,
    sep="\t",
    skiprows=header_line + 1,
    names=columns,
    comment="#",
    low_memory=False
)

# Selecting needed columns, they will be renamed in case we want to use other datasets
selected_cols = [
    "id.orig_p",    # src_port
    "id.resp_p",    # dst_port
    "proto",        # protocol (categorical)
    "duration",     # duration
    "orig_bytes",   # src_bytes
    "resp_bytes",   # dst_bytes
    "orig_pkts",    # src_pkts
    "resp_pkts",    # dst_pkts
    "conn_state",   # state (categorical)
    "label"         # label
]
selected_cols = [col for col in selected_cols if col in df.columns]
df = df[selected_cols]

rename_map = {
    "id.orig_p": "src_port",
    "id.resp_p": "dst_port",
    "orig_bytes": "src_bytes",
    "resp_bytes": "dst_bytes",
    "orig_pkts": "src_pkts",
    "resp_pkts": "dst_pkts",
    "conn_state": "state"
}
df = df.rename(columns=rename_map)

# Replace "-" with NaN
df.replace("-", np.nan, inplace=True)

# Convert numeric str columns to float
num_cols = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
cat_cols = ["proto", "state"]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaN in numeric columns with 0, categorical with 'unknown'
for col in num_cols:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

for col in cat_cols:
    if col in df.columns:
        df[col].fillna("unknown", inplace=True)

# Encode the label column as integers, 0: benign, 1: malicious
if "label" in df.columns:
    df["label_enc"], uniques = pd.factorize(df["label"])

# One-hot encode categorical features (proto and state)
df = pd.get_dummies(df, columns=cat_cols)

# Save the preprocessed data
df.to_csv(OUTPUT_CSV, index=False)

# Define feature columns and target variable
feature_cols = [
    'src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes', 
    'src_pkts', 'dst_pkts','proto_tcp', 'proto_udp', 'state_OTH', 
    'state_RSTR', 'state_S0', 'state_S1', 'state_S3', 'state_SF'
]

X = df[feature_cols]
y = df['label_enc']

print(X.head())
print(y.head())

