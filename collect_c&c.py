import pandas as pd
import numpy as np
import glob
import os
import argparse
import re
import tempfile


# default target if not provided via CLI
DEFAULT_TARGET_SAMPLES = 15000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pattern', type=str, default='**/*.labeled', help='glob pattern for .labeled files')
    p.add_argument('--chunked', action='store_true', help='use chunked processing for large files')
    p.add_argument('--upsample', action='store_true', help='allow upsampling when a label has fewer than target samples')
    p.add_argument('--target-samples', type=int, default=DEFAULT_TARGET_SAMPLES, help='max samples per non-C&C label')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    p.add_argument('--tmpdir', type=str, default=None, help='directory to use for temporary per-label files (will be created if missing). If not set, a tmp dir is created and removed after run.')
    p.add_argument('--keep-tmp', action='store_true', help='keep temporary files (do not delete tmpdir created by the script)')
    return p.parse_args()


args = parse_args()

# set runtime target from args
TARGET_SAMPLES = args.target_samples

"""This code will collect each C&C related sample from all datasets, which after there will be approximately 57k C&C samples.
It will also collect DDos, Benign, PortScan and Okiru samples from specific files to have a balanced dataset."""

# C&C-related labels to map to 'C&C'
cnc_labels = [
    "C&C",
    "C&C-FileDownload",
    "C&C-Torii",
    "C&C-HeartBeat",
    "C&C-HeartBeat-FileDownload",
    "C&C-PartOfAHorizontalPortScan",
    "C&C-HeartBeat-Attack"
]

LABEL_MAP = {lbl: "C&C" for lbl in cnc_labels}
LABEL_MAP.update({
    "Benign": "Benign",
    "DDoS": "DDoS",
    "PartOfAHorizontalPortScan": "PortScan",
    "Okiru": "Okiru"
})


def balanced_sample(df, label, n):
    subset = df[df['label'] == label]
    if len(subset) > n:
        return subset.sample(n=n, random_state=42)
    else:
        return subset

def parse_log(filename, selected_cols, rename_map):
    """Parse a .labeled file and return a cleaned DataFrame with selected/renamed columns."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    header_line_idx = next(i for i, line in enumerate(lines) if line.startswith("#fields"))
    raw_cols = lines[header_line_idx].strip().split("\t")[1:]
    # detect any header tokens that actually contain multiple logical names
    combined_indices = {}
    logical_columns = []
    for i, rc in enumerate(raw_cols):
        parts = re.split(r"\s+", rc.strip())
        if len(parts) > 1:
            combined_indices[i] = parts
            logical_columns.extend(parts)
        else:
            logical_columns.append(parts[0])

    data_rows = []
    for line in lines[header_line_idx + 1:]:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split('\t')
        # if a combined header existed, expand the matching data cell(s)
        if combined_indices and len(parts) == len(raw_cols):
            new_parts = []
            for idx, val in enumerate(parts):
                if idx in combined_indices:
                    # split the combined cell into the expected number of pieces
                    vals = re.split(r"\s+", val.strip())
                    # if splitting didn't yield the expected count, pad with empty strings
                    expected = len(combined_indices[idx])
                    if len(vals) < expected:
                        vals += [''] * (expected - len(vals))
                    new_parts.extend(vals[:expected])
                else:
                    new_parts.append(val)
            parts = new_parts

        if len(parts) != len(logical_columns):
            continue  # skip malformed lines

        row = dict(zip(logical_columns, parts))
        # Prefer `det_label` (more specific family like 'C&C') when available,
        # otherwise fall back to the coarse `label` field.
        det_label = row.get("det_label", None) or row.get("detailed-label", None) or row.get("detailed_label", None)
        if det_label and det_label != "-":
            chosen = det_label
        else:
            chosen = row.get("label", "")
        # map label if possible
        row['label'] = LABEL_MAP.get(chosen, chosen)
        data_rows.append(row)
    df = pd.DataFrame(data_rows)
    # Select and rename columns early for efficiency
    available_cols = [col for col in selected_cols if col in df.columns]
    df = df[available_cols]
    df = df.rename(columns=rename_map)
    return df

def clean_df(df):
    df = df.replace("-", np.nan)
    num_cols = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    cat_cols = ["proto", "state"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df


def chunked_process_file(path, selected_cols, rename_map, target_samples, upsample=False, seed=42, label_caps=None, tmpdir=None, keep_tmp=False):
    """Process a large .labeled file in chunks and return a concatenated DataFrame sampled per-label.

    Strategy:
    - Read header to get column names
    - Use pandas.read_csv with chunksize to iterate
    - Normalize label (prefer det_label)
    - Write rows for each label into temporary per-label csv files
    - After streaming, read each temp file and sample up to target_samples (with optional upsampling)
    - Concatenate sampled frames and return
    """
    import pandas as pd
    random_state = seed

    # prepare tmp dir (use provided tmpdir or create a transient one)
    created_tmpdir = False
    if tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
    else:
        tmpdir = tempfile.mkdtemp(prefix="collect_c_tmp_")
        created_tmpdir = True
    tmpfiles = {}  # label -> path

    # read header (#fields line) to get column names (robust to spaces inside fields)
    with open(path, 'r') as fh:
        for line in fh:
            if line.startswith("#fields"):
                raw_cols = line.strip().split("\t")[1:]
                # detect combined header tokens but don't expand them here
                combined_indices = {}
                cols = []
                for i, rc in enumerate(raw_cols):
                    parts = re.split(r"\s+", rc.strip())
                    if len(parts) > 1:
                        combined_indices[i] = parts
                        cols.append(rc)  # keep the original combined name for read_csv
                    else:
                        cols.append(parts[0])
                break
        else:
            raise RuntimeError(f"No #fields header found in {path}")

    # create pandas reader skipping comment lines
    reader = pd.read_csv(path, sep='\t', comment='#', names=cols, header=None, chunksize=200_000, low_memory=True)
    chunk_idx = 0
    debug_interval = 10  # print status every N chunks
    for chunk in reader:
        # if we read a column that actually contains multiple logical fields packed into one tab cell,
        # split that column into new columns now (e.g., last col like '-   Malicious   C&C')
        if 'combined_indices' in locals() and combined_indices:
            # iterate indices in descending order so insertion indices remain valid
            for idx in sorted(combined_indices.keys(), reverse=True):
                combined_name = raw_cols[idx]
                # the reader column name will be the original combined token (possibly with spaces)
                # find the actual column key in chunk.columns that matches by starting token
                # fallback: use the column at position idx
                try:
                    col_key = chunk.columns[idx]
                except Exception:
                    col_key = None
                if col_key is None or col_key not in chunk.columns:
                    continue
                parts_names = combined_indices[idx]
                # split string by whitespace into expected number of pieces
                split_df = chunk[col_key].astype(str).str.strip().str.split(r"\s+", n=len(parts_names)-1, expand=True)
                # ensure split_df has correct number of columns
                for j, new_name in enumerate(parts_names):
                    chunk[new_name] = split_df.iloc[:, j]
                # optionally drop the original combined column
                try:
                    chunk.drop(columns=[col_key], inplace=True)
                except Exception:
                    pass
        # prefer det_label when available, otherwise use label
        # prefer a detailed label column when available; accept multiple possible names
        detail_candidates = ['det_label', 'detailed-label', 'detailed_label', 'detailed label']
        det_col = None
        for c in detail_candidates:
            if c in chunk.columns:
                det_col = c
                break

        if det_col is not None:
            chosen = chunk[det_col].where(chunk[det_col] != '-', chunk.get('label'))
        else:
            chosen = chunk.get('label')

        # map known labels (e.g. map C&C variants to 'C&C') but keep original if unmapped
        if isinstance(chosen, pd.Series):
            chosen = chosen.map(LABEL_MAP).fillna(chosen)
            # strip whitespace for non-null strings
            non_null_mask = chosen.notnull()
            if non_null_mask.any():
                chosen.loc[non_null_mask] = chosen.loc[non_null_mask].astype(str).str.strip()
            # convert empty strings to NaN
            chosen = chosen.replace({'': np.nan, 'nan': np.nan})
        # assign back to the chunk and drop rows without a usable label
        chunk['chosen_label'] = chosen
        pre_count = len(chunk)
        chunk = chunk[chunk['chosen_label'].notna()]
        post_count = len(chunk)
        # debug / progress logging
        if chunk_idx % debug_interval == 0:
            try:
                top_labels = chunk['chosen_label'].value_counts().head(5).to_dict()
            except Exception:
                top_labels = {}
            print(f"[chunk {chunk_idx}] read={pre_count} kept={post_count} top_labels={top_labels}")
        chunk_idx += 1
        # select and rename columns we need (if present)
        available_cols = [c for c in selected_cols if c in chunk.columns]
        out_chunk = chunk[available_cols + ['chosen_label']].rename(columns=rename_map)
        # write per-label temp files
        for lbl, sub in out_chunk.groupby('chosen_label'):
            safe = lbl.replace('/', '_').replace(' ', '_')
            p = tmpfiles.get(safe)
            mode = 'a'
            header = False
            if p is None:
                p = os.path.join(tmpdir, f"tmp_{safe}.csv")
                tmpfiles[safe] = p
                header = True
            sub.to_csv(p, mode=mode, index=False, header=header)

    # now sample from each tmp file
    sampled_frames = []
    import math
    for safe, p in tmpfiles.items():
        df_lbl = pd.read_csv(p)
        lbl = df_lbl['chosen_label'].iloc[0] if 'chosen_label' in df_lbl.columns else safe
        n = len(df_lbl)
        if n == 0:
            continue
        # determine cap for this label: label_caps overrides target_samples
        cap = None
        if label_caps and lbl in label_caps:
            cap = label_caps[lbl]
        if cap is None:
            cap = target_samples

        if cap is None:
            # no cap -> keep all
            sampled = df_lbl
        else:
            if n > cap:
                sampled = df_lbl.sample(n=cap, random_state=random_state)
            else:
                if upsample and n > 0 and n < cap:
                    sampled = df_lbl.sample(n=cap, replace=True, random_state=random_state)
                else:
                    sampled = df_lbl
        sampled = sampled.drop(columns=['chosen_label'])
        sampled['label'] = lbl
        sampled_frames.append(sampled)

    if sampled_frames:
        out = pd.concat(sampled_frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=[c for c in rename_map.values()] + ['label'])

    # cleanup temporary files unless the caller asked to keep them
    if created_tmpdir and not keep_tmp:
        try:
            for f in os.listdir(tmpdir):
                try:
                    os.remove(os.path.join(tmpdir, f))
                except Exception:
                    pass
            os.rmdir(tmpdir)
        except Exception:
            pass

    return out


# Discover all .labeled files in the current directory and subdirectories
all_files = glob.glob(args.pattern, recursive=True)
# Fallback: look in current directory only
if not all_files:
    all_files = glob.glob('*.labeled')
if not all_files:
    raise FileNotFoundError("No .labeled files found. Place .labeled files in the workspace or adjust the glob pattern.")

# Columns to select and rename mapping
selected_cols = [
    "id.orig_p", "id.resp_p", "proto", "duration",
    "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
    "conn_state", "label"
]
rename_map = {
    "id.orig_p": "src_port",
    "id.resp_p": "dst_port",
    "orig_bytes": "src_bytes",
    "resp_bytes": "dst_bytes",
    "orig_pkts": "src_pkts",
    "resp_pkts": "dst_pkts",
    "conn_state": "state"
}

# For storing all rows
rows = []

for filename in all_files:
    print(f"Processing {filename}...")
    base = os.path.basename(filename)
    # Use chunked processing for very large files named like '43.1.labeled'
    if '43.1' in base:
        # label_caps: keep C&C unlimited (None), cap others to target
        label_caps = {"C&C": None}
        # other labels capped at target_samples
        df = chunked_process_file(filename, selected_cols, rename_map, target_samples=args.target_samples, upsample=args.upsample, seed=args.seed, label_caps=label_caps, tmpdir=args.tmpdir, keep_tmp=args.keep_tmp)
        # append C&C (all) and sampled others are already in df
        rows.append(df)
        continue


    # default small-file path
    df = parse_log(filename, selected_cols, rename_map)
    # Always collect all C&C samples
    rows.append(df[df['label'] == "C&C"])
    # Special handling for CTU-IoT-Malware-Capture-34-1 (Mirai)
    if "34.1" in base:
        for label in ["Benign", "DDoS", "PortScan"]:
            sampled = balanced_sample(df, label, args.target_samples)
            rows.append(sampled)
    # Special handling for CTU-IoT-Malware-Capture-17-1 (Kenjiro)
    if "17-1" in base:
        sampled = balanced_sample(df, "Okiru", args.target_samples)
        rows.append(sampled)


# Concatenate everything
df_all = pd.concat(rows, ignore_index=True)



# Save
df_all.to_csv("balanced_multiclass.csv", index=False)
print("✓ Saved balanced multiclass dataset to balanced_multiclass.csv")
print("Label distribution:")
print(df_all['label'].value_counts())
