import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os
import re
from config import MODEL_DIR, MAX_FEATURES, SEED

CATEGORICAL_COLS = [
    'message_type', 'stream_name', 'pipeline', 'protocol',
    'severity', 'action', 'product_name', 'vendor_name',
    'src_ip_first_octet', 'dst_ip_first_octet',
    'src_ip_class', 'dst_ip_class',
    'src_ip_version', 'dst_ip_version'
]

NUMERIC_COLS = [
    'hour', 'day_of_week', 'is_weekend', 'message_length',
    'has_ip', 'has_port', 'has_user', 'has_org',
    'src_ip_octet1', 'src_ip_octet2', 'src_ip_octet3', 'src_ip_octet4',
    'dst_ip_octet1', 'dst_ip_octet2', 'dst_ip_octet3', 'dst_ip_octet4'
]

MULTI_HOT_COLS = [
    'attack_techniques', 'attack_tactics', 'matched_rules', 'set_roles'
]

ENTITY_PATTERNS = [
    (r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', '<IP>'),
    (r'\b[a-fA-F0-9]{1,4}(?::[a-fA-F0-9]{1,4}){7}\b', '<IPV6>'),
    (r':\d{2,5}\b', '<PORT>'),
    (r'https?://[^\s]+', '<URL>'),
    (r'/\S+', '<PATH>'),
    (r'[\w.-]+@[\w.-]+\.\w+', '<EMAIL>'),
    (r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.[a-zA-Z]{2,}', '<DOMAIN>'),
    (r'\b[0-9a-fA-F]{32}\b|\b[0-9a-fA-F]{40}\b|\b[0-9a-fA-F]{64}\b', '<HASH>'),
    (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<GUID>'),
    (r'USER-\d+', '<USER>'),
    (r'ORG-\d+', '<ORG>'),
    (r'T\d{4}(\.\d+)?', '<ATTACK_TECHNIQUE>'),
    (r'TA\d{4}', '<ATTACK_TACTIC>'),
]

def normalize_entity(text):
    if not isinstance(text, str) or pd.isna(text):
        return ''
    for pattern, replacement in ENTITY_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

def extract_ip_features(df):
    df = df.copy()
    
    for prefix in ['src', 'dst']:
        ip_col = f'{prefix}_ip'
        
        df[f'{prefix}_ip_version'] = df[ip_col].apply(
            lambda x: 'ipv6' if isinstance(x, str) and ':' in x else 'ipv4'
        ).fillna('unknown')
        
        df[f'{prefix}_ip_class'] = df[ip_col].apply(classify_ip)
        
        df[[f'{prefix}_ip_octet1', f'{prefix}_ip_octet2', f'{prefix}_ip_octet3', f'{prefix}_ip_octet4']] = df[ip_col].apply(
            lambda x: pd.Series(parse_ip_to_octets(x))
        )
        
        df[f'{prefix}_ip_first_octet'] = df[f'{prefix}_ip_octet1'].astype(int).astype(str)
    
    return df

def parse_ip_to_octets(ip):
    if not isinstance(ip, str):
        return [0, 0, 0, 0]
    if ':' in ip:
        return [0, 0, 0, 0]
    
    parts = ip.split('.')
    octets = []
    for part in parts[:4]:
        try:
            octets.append(float(part))
        except ValueError:
            octets.append(0.0)
    
    while len(octets) < 4:
        octets.append(0.0)
    
    return octets

def classify_ip(ip):
    if not isinstance(ip, str):
        return 'unknown'
    
    if ':' in ip:
        if ip.startswith('fe80:'):
            return 'ipv6_link_local'
        elif ip.startswith('fc00:') or ip.startswith('fd00:'):
            return 'ipv6_unique_local'
        else:
            return 'ipv6_public'
    
    try:
        parts = list(map(int, ip.split('.')))
        if len(parts) == 4:
            if parts[0] == 10:
                return 'private_a'
            elif parts[0] == 172 and 16 <= parts[1] <= 31:
                return 'private_b'
            elif parts[0] == 192 and parts[1] == 168:
                return 'private_c'
            elif parts[0] == 127:
                return 'loopback'
            elif parts[0] == 169 and parts[1] == 254:
                return 'link_local'
            elif parts[0] == 100 and 64 <= parts[1] <= 127:
                return 'carrier_grade_nat'
            else:
                return 'public'
    except:
        pass
    
    return 'unknown'

def parse_multi_value_field(value):
    if pd.isna(value) or value in ['None', 'none', '']:
        return []
    if isinstance(value, str):
        values = [v.strip() for v in value.split(',') if v.strip()]
        return values
    return []

class FeatureEngineer:
    def __init__(self):
        self.tfidf = None
        self.scaler = None
        self.onehot_encoders = {}
        self.multi_hot_vocabularies = {}
    
    def fit_text_features(self, texts):
        normalized_texts = texts.apply(normalize_entity)
        self.tfidf = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\w+'
        )
        self.tfidf.fit(normalized_texts)
        return self
    
    def transform_text_features(self, texts):
        normalized_texts = texts.apply(normalize_entity)
        return self.tfidf.transform(normalized_texts)
    
    def fit_categorical_features(self, df):
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('unknown')
                unique_vals = df[col].unique()
                if 'unknown' not in unique_vals:
                    unique_vals = np.append(unique_vals, 'unknown')
                ohe = OneHotEncoder(categories=[unique_vals], handle_unknown='ignore', sparse_output=True)
                ohe.fit(df[[col]])
                self.onehot_encoders[col] = ohe
        return self
    
    def transform_categorical_features(self, df):
        df = df.copy()
        encoded_features = []
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.onehot_encoders:
                df[col] = df[col].astype(str).fillna('unknown')
                encoded = self.onehot_encoders[col].transform(df[[col]])
                encoded_features.append(encoded)
        
        if encoded_features:
            return sparse_hstack(encoded_features)
        return csr_matrix(np.zeros((len(df), 0)))
    
    def fit_multi_hot_features(self, df):
        for col in MULTI_HOT_COLS:
            if col in df.columns:
                all_values = set()
                df[col].apply(lambda x: all_values.update(parse_multi_value_field(x)))
                vocab = sorted(list(all_values))
                if vocab:
                    self.multi_hot_vocabularies[col] = {v: i for i, v in enumerate(vocab)}
        return self
    
    def transform_multi_hot_features(self, df):
        all_features = []
        for col in MULTI_HOT_COLS:
            if col in df.columns and col in self.multi_hot_vocabularies:
                vocab = self.multi_hot_vocabularies[col]
                num_features = len(vocab)
                feature_matrix = np.zeros((len(df), num_features), dtype=np.float32)
                
                for idx, value in enumerate(df[col]):
                    values = parse_multi_value_field(value)
                    for v in values:
                        if v in vocab:
                            feature_matrix[idx, vocab[v]] = 1.0
                
                all_features.append(csr_matrix(feature_matrix))
        
        if all_features:
            return sparse_hstack(all_features)
        return csr_matrix(np.zeros((len(df), 0)))
    
    def fit_numeric_features(self, df):
        available_numeric = [c for c in NUMERIC_COLS if c in df.columns]
        self.scaler = StandardScaler()
        numeric_data = df[available_numeric].fillna(0).values
        self.scaler.fit(numeric_data)
        self.scaler.feature_names = available_numeric
        return self
    
    def transform_numeric_features(self, df):
        available_numeric = self.scaler.feature_names if hasattr(self.scaler, 'feature_names') else NUMERIC_COLS
        available_numeric = [c for c in available_numeric if c in df.columns]
        numeric_data = df[available_numeric].fillna(0).values
        return csr_matrix(self.scaler.transform(numeric_data))
    
    def fit(self, df):
        df = extract_ip_features(df)
        self.fit_text_features(df['message_sanitized'])
        self.fit_categorical_features(df)
        self.fit_multi_hot_features(df)
        self.fit_numeric_features(df)
        return self
    
    def transform(self, df):
        df = extract_ip_features(df)
        text_features = self.transform_text_features(df['message_sanitized'])
        cat_features = self.transform_categorical_features(df)
        multi_hot_features = self.transform_multi_hot_features(df)
        num_features = self.transform_numeric_features(df)
        
        all_features = [text_features, cat_features, multi_hot_features, num_features]
        return sparse_hstack([f for f in all_features if f.shape[1] > 0])
    
    def save(self, path=None):
        if path is None:
            path = MODEL_DIR
        
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.tfidf, os.path.join(path, 'tfidf.pkl'))
        joblib.dump(self.onehot_encoders, os.path.join(path, 'onehot_encoders.pkl'))
        joblib.dump(self.multi_hot_vocabularies, os.path.join(path, 'multi_hot_vocabularies.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
    
    @classmethod
    def load(cls, path=None):
        if path is None:
            path = MODEL_DIR
        
        fe = cls()
        fe.tfidf = joblib.load(os.path.join(path, 'tfidf.pkl'))
        fe.onehot_encoders = joblib.load(os.path.join(path, 'onehot_encoders.pkl'))
        fe.multi_hot_vocabularies = joblib.load(os.path.join(path, 'multi_hot_vocabularies.pkl'))
        fe.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        return fe

def extract_features(df, feature_engineer=None, fit=False):
    if fit:
        fe = FeatureEngineer()
        fe.fit(df)
        features = fe.transform(df)
        return features, fe
    else:
        return feature_engineer.transform(df)