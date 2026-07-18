import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from data_loader import load_train_data, preprocess_data
from pipeline import FeaturePipeline
from model_trainer import train_and_evaluate, ModelTrainer
from sklearn.model_selection import StratifiedKFold, train_test_split
import time

LEAKAGE_COLS = [
    'label_binary', 'lifecycle_stage', 'mo_name', 'matched_rules', 
    'set_roles', 'attack_techniques', 'attack_tactics', 'event_id',
    'severity', 'action', 'product_name', 'vendor_name', 'vendor_code',
    'username', 'src_ip', 'dst_ip', 'src_host', 'dst_host',
    'stream_name', 'message_type', 'message_sanitized',
    'label_confidence', 'suspicion_score', 'pipeline'
]

SEEDS = [5, 42, 2025, 3407, 114514]

def run_validation_with_seed(seed):
    print(f"\n{'='*60}")
    print(f"种子 {seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    train_df = load_train_data()
    train_df = preprocess_data(train_df, is_train=True)
    
    sample_size = 50000
    train_df_cv = train_df.groupby('label_binary', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 3), random_state=seed)
    ).reset_index(drop=True)
    
    print(f"采样后数据形状: {train_df_cv.shape}")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_cv, train_df_cv['label'])):
        train_fold = train_df_cv.iloc[train_idx]
        val_fold = train_df_cv.iloc[val_idx]
        
        train_fold_clean = train_fold.drop(columns=[c for c in LEAKAGE_COLS if c in train_fold.columns])
        val_fold_clean = val_fold.drop(columns=[c for c in LEAKAGE_COLS if c in val_fold.columns])
        
        fe = FeaturePipeline()
        fe.fit(train_fold_clean)
        
        X_train = fe.transform(train_fold_clean)
        X_val = fe.transform(val_fold_clean)
        y_train = train_fold['label'].values
        y_val = val_fold['label'].values
        
        trainer, metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val,
            model_type='lgb'
        )
        
        fold_metrics.append(metrics)
        print(f"Fold {fold+1}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'f1_macro': np.mean([m['f1_macro'] for m in fold_metrics]),
        'recall_macro': np.mean([m['recall_macro'] for m in fold_metrics]),
        'malicious_recall': np.mean([m['malicious_recall'] for m in fold_metrics]),
        'suspicious_recall': np.mean([m['suspicious_recall'] for m in fold_metrics]),
        'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
        'std_f1_macro': np.std([m['f1_macro'] for m in fold_metrics]),
    }
    
    print(f"\n平均指标:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return avg_metrics

def main():
    print("多随机种子验证")
    print(f"验证种子: {SEEDS}")
    
    results = []
    
    for seed in SEEDS:
        start_time = time.time()
        metrics = run_validation_with_seed(seed)
        elapsed = time.time() - start_time
        metrics['seed'] = seed
        metrics['time_seconds'] = elapsed
        results.append(metrics)
    
    print(f"\n{'='*60}")
    print("多种子验证汇总")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print(f"\n统计摘要:")
    print(f"平均准确率: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"平均F1: {results_df['f1_macro'].mean():.4f} ± {results_df['f1_macro'].std():.4f}")
    print(f"最小准确率: {results_df['accuracy'].min():.4f}")
    print(f"最大准确率: {results_df['accuracy'].max():.4f}")
    
    results_df.to_csv('logs/multi_seed_validation.csv', index=False)
    print("\n结果已保存至 logs/multi_seed_validation.csv")

if __name__ == '__main__':
    main()
