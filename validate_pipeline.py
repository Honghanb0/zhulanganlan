import time
import warnings
import json
import datetime
import os

warnings.filterwarnings('ignore')

from data_loader import load_train_data, preprocess_data
from pipeline import FeaturePipeline, CATEGORICAL_COLS, NUMERIC_COLS, MULTI_HOT_COLS
from model_trainer import train_and_evaluate, ModelTrainer, evaluate_model
from predictor import predict_on_test_data
from config import VALID_FILE, SEED, CLASS_WEIGHTS, MODEL_DIR, MAX_FEATURES, REVERSE_LABEL_MAPPING
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def save_metadata(metadata, path=None):
    if path is None:
        path = MODEL_DIR
    os.makedirs(path, exist_ok=True)
    metadata_path = os.path.join(path, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"Metadata saved to {metadata_path}")

def main():
    print("=" * 60)
    print("Pipeline Validation - Full Workflow Test")
    print("=" * 60)
    
    start_time = time.time()
    start_timestamp = datetime.datetime.now().isoformat()
    
    print("\n[1/7] Loading training data...")
    train_df = load_train_data()
    print(f"Training data shape: {train_df.shape}")
    print(f"Label distribution:\n{train_df['label_binary'].value_counts()}")
    
    print("\n[2/7] Preprocessing data...")
    train_df = preprocess_data(train_df, is_train=True)
    
    print("\n[3/7] Running 5-fold StratifiedKFold cross validation (sampled)...")
    cv_sample_size = 50000
    train_df_cv = train_df.groupby('label_binary', group_keys=False).apply(
        lambda x: x.sample(min(len(x), cv_sample_size // 3), random_state=SEED)
    ).reset_index(drop=True)
    print(f"CV data shape: {train_df_cv.shape}")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    fe_base = FeaturePipeline()
    fe_base.fit(train_df_cv)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_cv, train_df_cv['label'])):
        train_fold = train_df_cv.iloc[train_idx]
        val_fold = train_df_cv.iloc[val_idx]
        
        print(f"\nFold {fold+1}/5: train={len(train_fold)}, val={len(val_fold)}")
        
        fe = FeaturePipeline()
        fe.tfidf = fe_base.tfidf
        fe.onehot_encoders = fe_base.onehot_encoders
        fe.multi_hot_encoder = fe_base.multi_hot_encoder
        fe.scaler = fe_base.scaler
        fe.ip_extractor = fe_base.ip_extractor
        fe.entity_normalizer = fe_base.entity_normalizer
        fe.categorical_cols = fe_base.categorical_cols
        fe.numeric_cols = fe_base.numeric_cols
        fe.seed = fe_base.seed
        
        X_train = fe.transform(train_fold)
        X_val = fe.transform(val_fold)
        y_train = train_fold['label'].values
        y_val = val_fold['label'].values
        
        print(f"Feature shape: {X_train.shape}")
        
        trainer, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, model_type='lgb')
        fold_metrics.append(metrics)
        
        y_pred = trainer.predict(X_val)
        y_proba = trainer.predict_proba(X_val)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'f1_macro': np.mean([m['f1_macro'] for m in fold_metrics]),
        'recall_macro': np.mean([m['recall_macro'] for m in fold_metrics]),
        'malicious_recall': np.mean([m['malicious_recall'] for m in fold_metrics]),
        'suspicious_recall': np.mean([m['suspicious_recall'] for m in fold_metrics]),
        'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
        'std_f1_macro': np.std([m['f1_macro'] for m in fold_metrics]),
    }
    
    print("\n=== Cross Validation Results ===")
    for i, m in enumerate(fold_metrics):
        print(f"\nFold {i+1}:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    
    print("\n=== CV Summary ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
    
    stability_threshold = 0.15
    acc_ratio = avg_metrics['std_accuracy'] / avg_metrics['accuracy'] if avg_metrics['accuracy'] > 0 else 1
    f1_ratio = avg_metrics['std_f1_macro'] / avg_metrics['f1_macro'] if avg_metrics['f1_macro'] > 0 else 1
    
    print(f"\nStability Check:")
    print(f"  Accuracy std/mean: {acc_ratio:.4f} (threshold: {stability_threshold})")
    print(f"  F1 std/mean: {f1_ratio:.4f} (threshold: {stability_threshold})")
    
    if acc_ratio <= stability_threshold and f1_ratio <= stability_threshold:
        print("  ✅ Stability check PASSED")
    else:
        print("  ⚠️ Stability check WARNING")
    
    print("\n[4/7] Training model on full data (small sample for speed)...")
    sample_size = 50000
    train_df_sample = train_df.groupby('label_binary', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 3), random_state=SEED)
    ).reset_index(drop=True)
    
    fe_final = FeaturePipeline()
    fe_final.fit(train_df_sample)
    X_final = fe_final.transform(train_df_sample)
    y_final = train_df_sample['label'].values
    print(f"Feature shape: {X_final.shape}")
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=SEED, stratify=y_final)
    trainer_final, _ = train_and_evaluate(X_tr, y_tr, X_val, y_val, model_type='lgb')
    
    print("Saving model and pipeline...")
    fe_final.save()
    trainer_final.save()
    
    print("\n[5/7] Testing inference...")
    valid_df = pd.read_parquet(VALID_FILE)
    valid_df = preprocess_data(valid_df, is_train=False)
    print(f"Test data shape: {valid_df.shape}")
    
    X_test = fe_final.transform(valid_df)
    print(f"Test feature shape: {X_test.shape}")
    print(f"Train feature shape: {X_final.shape}")
    
    if X_test.shape[1] != X_final.shape[1]:
        print(f"❌ Feature dimension mismatch! Train: {X_final.shape[1]}, Test: {X_test.shape[1]}")
    else:
        print("✅ Feature dimension matches")
    
    infer_start = time.time()
    y_pred = trainer_final.predict(X_test)
    infer_time = time.time() - infer_start
    print(f"Inference time: {infer_time:.2f}s")
    print(f"Inference speed: {len(valid_df)/infer_time:.1f} samples/sec")
    
    result_df = pd.DataFrame({
        'event_id': valid_df['event_id'],
        'pred_label': [REVERSE_LABEL_MAPPING[int(pred)] for pred in y_pred]
    })
    print(f"Prediction distribution:")
    print(result_df['pred_label'].value_counts())
    
    print("\n[6/7] Validating metadata...")
    metadata = {
        'version': '2.0',
        'train_start_time': start_timestamp,
        'train_end_time': datetime.datetime.now().isoformat(),
        'duration_seconds': round(time.time() - start_time, 2),
        'seed': SEED,
        'num_folds': 5,
        'data': {
            'total_samples': len(train_df),
            'cv_samples': len(train_df_cv),
            'label_distribution': train_df['label_binary'].value_counts().to_dict(),
            'feature_shape': str(X_final.shape),
        },
        'features': {
            'categorical_cols': CATEGORICAL_COLS,
            'numeric_cols': NUMERIC_COLS,
            'multi_hot_cols': MULTI_HOT_COLS,
            'text_col': 'message_sanitized',
            'max_features': MAX_FEATURES,
        },
        'model': {
            'type': 'lightgbm',
            'class_weights': CLASS_WEIGHTS,
            'default_params': {
                'objective': 'multiclass',
                'num_class': 3,
                'boosting_type': 'gbdt',
                'num_leaves': 256,
                'learning_rate': 0.03,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'random_state': SEED,
                'is_unbalance': True,
                'min_data_in_leaf': 30,
                'max_depth': -1,
                'reg_alpha': 0.05,
                'reg_lambda': 0.05,
                'n_estimators': 1000,
            },
        },
        'cross_validation': {
            'avg_metrics': avg_metrics,
            'fold_metrics': fold_metrics,
            'stability_check': {
                'accuracy_ratio': acc_ratio,
                'f1_ratio': f1_ratio,
                'threshold': stability_threshold,
                'passed': acc_ratio <= stability_threshold and f1_ratio <= stability_threshold,
            },
        },
        'inference': {
            'samples': len(result_df),
            'time_seconds': round(infer_time, 2),
            'speed_samples_per_sec': round(len(valid_df)/infer_time, 1),
            'prediction_distribution': result_df['pred_label'].value_counts().to_dict(),
        },
        'entity_patterns': [
            {'pattern': 'IPv4 address', 'replacement': '<IP>'},
            {'pattern': 'IPv6 address', 'replacement': '<IPV6>'},
            {'pattern': 'Port number', 'replacement': '<PORT>'},
            {'pattern': 'URL', 'replacement': '<URL>'},
            {'pattern': 'File path', 'replacement': '<PATH>'},
            {'pattern': 'Email', 'replacement': '<EMAIL>'},
            {'pattern': 'Domain', 'replacement': '<DOMAIN>'},
            {'pattern': 'Hash', 'replacement': '<HASH>'},
            {'pattern': 'GUID', 'replacement': '<GUID>'},
            {'pattern': 'User ID', 'replacement': '<USER>'},
            {'pattern': 'Org ID', 'replacement': '<ORG>'},
            {'pattern': 'MITRE ATT&CK Technique', 'replacement': '<ATTACK_TECHNIQUE>'},
            {'pattern': 'MITRE ATT&CK Tactic', 'replacement': '<ATTACK_TACTIC>'},
        ],
    }
    
    save_metadata(metadata)
    
    print("\n[7/7] Verifying metadata integrity...")
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'r', encoding='utf-8') as f:
        loaded_metadata = json.load(f)
    
    required_fields = ['version', 'seed', 'num_folds', 'data', 'features', 'model', 'cross_validation', 'inference']
    missing_fields = [f for f in required_fields if f not in loaded_metadata]
    
    if not missing_fields:
        print("✅ All required metadata fields present")
    else:
        print(f"❌ Missing fields: {missing_fields}")
    
    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    return avg_metrics

if __name__ == '__main__':
    main()