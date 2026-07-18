import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack as sparse_hstack, csr_matrix
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

def parse_multi_value_field(value):
    if pd.isna(value) or value in ['None', 'none', '']:
        return []
    if isinstance(value, str):
        values = [v.strip() for v in value.split(',') if v.strip()]
        return values
    return []

def parse_ip_octets_vectorized(ip_series):
    result = np.zeros((len(ip_series), 4), dtype=np.float64)
    for i, ip in enumerate(ip_series):
        if isinstance(ip, str):
            if ':' not in ip:
                parts = ip.split('.')
                for j, part in enumerate(parts[:4]):
                    try:
                        result[i, j] = float(part)
                    except ValueError:
                        result[i, j] = 0.0
    return result

class IPFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for prefix in ['src', 'dst']:
            ip_col = f'{prefix}_ip'
            if ip_col in X.columns:
                ip_series = X[ip_col].fillna('')
                X[f'{prefix}_ip_version'] = np.where(
                    ip_series.str.contains(':', na=False), 'ipv6', 'ipv4'
                )
                X[f'{prefix}_ip_class'] = ip_series.apply(classify_ip)
                
                octets = parse_ip_octets_vectorized(ip_series.values)
                X[f'{prefix}_ip_octet1'] = octets[:, 0]
                X[f'{prefix}_ip_octet2'] = octets[:, 1]
                X[f'{prefix}_ip_octet3'] = octets[:, 2]
                X[f'{prefix}_ip_octet4'] = octets[:, 3]
                
                X[f'{prefix}_ip_first_octet'] = X[f'{prefix}_ip_octet1'].astype(int).astype(str)
        return X

class EntityNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'message_sanitized' in X.columns:
            X['message_sanitized'] = X['message_sanitized'].apply(normalize_entity)
        return X

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=MULTI_HOT_COLS):
        self.cols = cols
        self.vocabularies = {}
    
    def fit(self, X, y=None):
        for col in self.cols:
            if col in X.columns:
                all_values = set()
                X[col].apply(lambda x: all_values.update(parse_multi_value_field(x)))
                vocab = sorted(list(all_values))
                if vocab:
                    self.vocabularies[col] = {v: i for i, v in enumerate(vocab)}
        return self
    
    def transform(self, X):
        all_features = []
        for col in self.cols:
            if col in self.vocabularies:
                vocab = self.vocabularies[col]
                num_features = len(vocab)
                feature_matrix = np.zeros((len(X), num_features), dtype=np.float32)
                
                if col in X.columns:
                    for idx, value in enumerate(X[col]):
                        values = parse_multi_value_field(value)
                        for v in values:
                            if v in vocab:
                                feature_matrix[idx, vocab[v]] = 1.0
                
                all_features.append(csr_matrix(feature_matrix))
        
        if all_features:
            return sparse_hstack(all_features)
        return csr_matrix(np.zeros((len(X), 0)))

class FeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, seed=SEED):
        self.seed = seed
        self.ip_extractor = IPFeatureExtractor()
        self.entity_normalizer = EntityNormalizer()
        self.tfidf = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\w+'
        )
        self.onehot_encoders = {}
        self.scaler = StandardScaler()
        self.multi_hot_encoder = MultiHotEncoder(MULTI_HOT_COLS)
        self.categorical_cols = CATEGORICAL_COLS
        self.numeric_cols = NUMERIC_COLS
    
    def fit(self, X, y=None):
        np.random.seed(self.seed)
        X = X.copy()
        X = self.ip_extractor.fit_transform(X)
        X = self.entity_normalizer.fit_transform(X)
        
        if 'message_sanitized' in X.columns:
            self.tfidf.fit(X['message_sanitized'])
        self._has_text = 'message_sanitized' in X.columns
        
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('unknown')
                unique_vals = X[col].unique()
                if 'unknown' not in unique_vals:
                    unique_vals = np.append(unique_vals, 'unknown')
                ohe = OneHotEncoder(categories=[unique_vals], handle_unknown='ignore', sparse_output=True)
                ohe.fit(X[[col]])
                self.onehot_encoders[col] = ohe
        
        available_multi_hot = [c for c in MULTI_HOT_COLS if c in X.columns]
        if available_multi_hot:
            self.multi_hot_encoder.fit(X)
        self._has_multi_hot = len(available_multi_hot) > 0
        
        available_numeric = [c for c in self.numeric_cols if c in X.columns]
        if available_numeric:
            self.scaler.fit(X[available_numeric].fillna(0).values)
            self.scaler.feature_names = available_numeric
        
        return self
    
    def transform(self, X):
        X = X.copy()
        X = self.ip_extractor.transform(X)
        X = self.entity_normalizer.transform(X)
        
        if hasattr(self, '_has_text') and self._has_text and 'message_sanitized' in X.columns:
            text_features = self.tfidf.transform(X['message_sanitized'])
        else:
            text_features = csr_matrix(np.zeros((len(X), 0)))
        
        cat_features_list = []
        for col in self.categorical_cols:
            if col in self.onehot_encoders:
                if col in X.columns:
                    X[col] = X[col].astype(str).fillna('unknown')
                else:
                    X[col] = 'unknown'
                encoded = self.onehot_encoders[col].transform(X[[col]])
                cat_features_list.append(encoded)
        
        cat_features = sparse_hstack(cat_features_list) if cat_features_list else csr_matrix(np.zeros((len(X), 0)))
        
        if hasattr(self, '_has_multi_hot') and self._has_multi_hot:
            multi_hot_features = self.multi_hot_encoder.transform(X)
        else:
            multi_hot_features = csr_matrix(np.zeros((len(X), 0)))
        
        available_numeric = getattr(self.scaler, 'feature_names', self.numeric_cols)
        numeric_data = []
        for c in available_numeric:
            if c in X.columns:
                numeric_data.append(X[c].fillna(0).values.reshape(-1, 1))
            else:
                numeric_data.append(np.zeros((len(X), 1)))
        if numeric_data:
            num_array = np.hstack(numeric_data)
            num_features = csr_matrix(self.scaler.transform(num_array))
        else:
            num_features = csr_matrix(np.zeros((len(X), 0)))
        
        all_features = [text_features, cat_features, multi_hot_features, num_features]
        return sparse_hstack([f for f in all_features if f.shape[1] > 0])
    
    def save(self, path=None):
        if path is None:
            path = MODEL_DIR
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.tfidf, os.path.join(path, 'tfidf.pkl'))
        joblib.dump(self.onehot_encoders, os.path.join(path, 'onehot_encoders.pkl'))
        joblib.dump(self.multi_hot_encoder, os.path.join(path, 'multi_hot_encoder.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(self.ip_extractor, os.path.join(path, 'ip_extractor.pkl'))
        joblib.dump(self.entity_normalizer, os.path.join(path, 'entity_normalizer.pkl'))
    
    @classmethod
    def load(cls, path=None):
        if path is None:
            path = MODEL_DIR
        pipeline = cls()
        pipeline.tfidf = joblib.load(os.path.join(path, 'tfidf.pkl'))
        pipeline.onehot_encoders = joblib.load(os.path.join(path, 'onehot_encoders.pkl'))
        pipeline.multi_hot_encoder = joblib.load(os.path.join(path, 'multi_hot_encoder.pkl'))
        pipeline.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        pipeline.ip_extractor = joblib.load(os.path.join(path, 'ip_extractor.pkl'))
        pipeline.entity_normalizer = joblib.load(os.path.join(path, 'entity_normalizer.pkl'))
        return pipeline