import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from config import TRAIN_FILE, VALID_FILE, LABEL_MAPPING, SEED, SAMPLE_SIZE

def load_train_data(sample_size=None):
    sample_size = sample_size or SAMPLE_SIZE
    
    parquet_file = pq.ParquetFile(TRAIN_FILE)
    total_rows = parquet_file.metadata.num_rows
    
    if total_rows <= sample_size:
        df = parquet_file.read().to_pandas()
    else:
        num_row_groups = parquet_file.num_row_groups
        rows_per_group = []
        for i in range(num_row_groups):
            rows_per_group.append(parquet_file.metadata.row_group(i).num_rows)
        
        cumulative_rows = np.cumsum(rows_per_group)
        target_rows_per_class = sample_size // 3
        
        sampled_rows = []
        current_counts = {'benign': 0, 'malicious': 0, 'suspicious': 0}
        
        np.random.seed(SEED)
        group_order = np.random.permutation(num_row_groups)
        
        for group_idx in group_order:
            table = parquet_file.read_row_group(group_idx)
            df_group = table.to_pandas()
            
            for label in ['benign', 'malicious', 'suspicious']:
                if current_counts[label] >= target_rows_per_class:
                    continue
                mask = df_group['label_binary'] == label
                remaining = target_rows_per_class - current_counts[label]
                df_label = df_group[mask]
                if len(df_label) > remaining:
                    df_label = df_label.sample(n=remaining, random_state=SEED)
                sampled_rows.append(df_label)
                current_counts[label] += len(df_label)
            
            if all(c >= target_rows_per_class for c in current_counts.values()):
                break
        
        df = pd.concat(sampled_rows, ignore_index=True)
    
    df['label'] = df['label_binary'].map(LABEL_MAPPING)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df

def load_valid_data():
    return pd.read_parquet(VALID_FILE)

def preprocess_data(df, is_train=True):
    df = df.copy()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    for col in ['src_port', 'dst_port']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({'': '-1', 'None': '-1'}).fillna('-1').astype(int)
        else:
            df[col] = -1
    
    for col in ['src_ip', 'dst_ip', 'username', 'action', 'severity', 'vendor_code']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = 'unknown'
    
    df['message_sanitized'] = df['message_sanitized'].fillna('') if 'message_sanitized' in df.columns else ''
    
    df['message_length'] = df['message_sanitized'].str.len() if 'message_sanitized' in df.columns else 0
    df['has_ip'] = df['message_sanitized'].str.contains(r'\d+\.\d+\.\d+\.\d+').fillna(False).astype(int) if 'message_sanitized' in df.columns else 0
    df['has_port'] = df['message_sanitized'].str.contains(r':\d{2,5}').fillna(False).astype(int) if 'message_sanitized' in df.columns else 0
    df['has_user'] = df['message_sanitized'].str.contains(r'USER-\d+').fillna(False).astype(int) if 'message_sanitized' in df.columns else 0
    df['has_org'] = df['message_sanitized'].str.contains(r'ORG-\d+').fillna(False).astype(int) if 'message_sanitized' in df.columns else 0
    
    for col in ['attack_techniques', 'attack_tactics', 'matched_rules', 'set_roles']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    if is_train:
        df = df.dropna(subset=['label'])
    
    return df

def split_data(df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df['label'])
    return train_df, val_df