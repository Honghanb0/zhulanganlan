import pandas as pd
import numpy as np
from config import TRAIN_FILE, VALID_FILE, LABEL_MAPPING, SEED, SAMPLE_SIZE

def load_train_data(sample_size=None):
    df = pd.read_parquet(TRAIN_FILE)
    df['label'] = df['label_binary'].map(LABEL_MAPPING)
    
    sample_size = sample_size or SAMPLE_SIZE
    if len(df) > sample_size:
        df = df.groupby('label_binary', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 3), random_state=SEED)
        ).reset_index(drop=True)
    
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