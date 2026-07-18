import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore')

from data_loader import load_train_data, preprocess_data
from pipeline import FeaturePipeline, CATEGORICAL_COLS, NUMERIC_COLS, MULTI_HOT_COLS
from model_trainer import train_and_evaluate, ModelTrainer, evaluate_model
from predictor import predict_on_test_data
from config import VALID_FILE, SEED, CLASS_WEIGHTS, MODEL_DIR, MAX_FEATURES
from train_runner import create_runner
from lgbm_callbacks import create_lgbm_callbacks, create_early_stopping_callback

def run_training_pipeline(runner):
    logger = runner.logger
    tracker = runner.tracker
    
    with tracker.stage('数据加载'):
        logger.info("加载训练数据...")
        train_df = load_train_data()
        logger.info(f"训练数据形状: {train_df.shape}")
        logger.info(f"标签分布:\n{train_df['label_binary'].value_counts()}")
    
    with tracker.stage('数据预处理'):
        logger.info("预处理数据...")
        train_df = preprocess_data(train_df, is_train=True)
    
    with tracker.stage('数据采样'):
        train_df_cv = train_df.copy()
        logger.info(f"训练数据形状: {train_df_cv.shape}")
        logger.info(f"标签分布:\n{train_df_cv['label_binary'].value_counts()}")
    
    runner.record_data_info({
        'total_samples': len(train_df),
        'cv_samples': len(train_df_cv),
        'label_distribution': train_df['label_binary'].value_counts().to_dict()
    })
    
    LEAKAGE_COLS = [
        'label_binary', 'lifecycle_stage', 'mo_name', 'matched_rules', 
        'set_roles', 'attack_techniques', 'attack_tactics', 'event_id',
        'severity', 'action', 'product_name', 'vendor_name', 'vendor_code',
        'username', 'src_ip', 'dst_ip', 'src_host', 'dst_host',
        'stream_name', 'message_type', 'message_sanitized',
        'label_confidence', 'suspicion_score', 'pipeline'
    ]
    
    with tracker.stage('交叉验证'):
        logger.info("运行5折StratifiedKFold交叉验证...")
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df_cv, train_df_cv['label'])):
            with tracker.stage(f'交叉验证_Fold{fold+1}'):
                logger.info(f"Fold {fold + 1}/5")
                
                train_fold = train_df_cv.iloc[train_idx]
                val_fold = train_df_cv.iloc[val_idx]
                
                train_fold_clean = train_fold.drop(columns=[c for c in LEAKAGE_COLS if c in train_fold.columns])
                val_fold_clean = val_fold.drop(columns=[c for c in LEAKAGE_COLS if c in val_fold.columns])
                
                logger.info(f"训练样本: {len(train_fold)}, 验证样本: {len(val_fold)}")
                logger.info(f"移除泄漏特征后特征数: {len(train_fold_clean.columns)}")
                
                fe = FeaturePipeline()
                fe.fit(train_fold_clean)
                
                X_train = fe.transform(train_fold_clean)
                X_val = fe.transform(val_fold_clean)
                y_train = train_fold['label'].values
                y_val = val_fold['label'].values
                
                logger.info(f"特征维度: {X_train.shape}")
                
                callbacks = create_lgbm_callbacks(logger, log_interval=100)
                early_stopping = create_early_stopping_callback(logger, patience=100)
                
                trainer, metrics = train_and_evaluate(
                    X_train, y_train, X_val, y_val, 
                    model_type='lgb',
                    callbacks=callbacks + [early_stopping]
                )
                
                fold_metrics.append(metrics)
                
                y_pred = trainer.predict(X_val)
                y_proba = trainer.predict_proba(X_val)
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred)
                all_y_proba.extend(y_proba)
                
                logger.info(f"Fold {fold + 1} 完成 - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'f1_macro': np.mean([m['f1_macro'] for m in fold_metrics]),
        'recall_macro': np.mean([m['recall_macro'] for m in fold_metrics]),
        'malicious_recall': np.mean([m['malicious_recall'] for m in fold_metrics]),
        'suspicious_recall': np.mean([m['suspicious_recall'] for m in fold_metrics]),
        'std_accuracy': np.std([m['accuracy'] for m in fold_metrics]),
        'std_f1_macro': np.std([m['f1_macro'] for m in fold_metrics]),
    }
    
    overall_metrics = evaluate_model(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba))
    
    runner.record_cv_results({
        'avg_metrics': avg_metrics,
        'fold_metrics': fold_metrics,
        'overall_metrics': overall_metrics
    })
    
    logger.info("交叉验证结果汇总:")
    for key, value in avg_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    with tracker.stage('最终模型训练'):
        logger.info("在完整训练数据上训练最终模型...")
        
        sample_size_final = 50000
        train_df_sample = train_df.groupby('label_binary', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size_final // 3), random_state=SEED)
        ).reset_index(drop=True)
        
        train_df_sample_clean = train_df_sample.drop(columns=[c for c in LEAKAGE_COLS if c in train_df_sample.columns])
        
        logger.info(f"训练样本数: {len(train_df_sample)}")
        logger.info(f"移除泄漏特征后特征数: {len(train_df_sample_clean.columns)}")
        
        fe_final = FeaturePipeline()
        fe_final.fit(train_df_sample_clean)
        X_final = fe_final.transform(train_df_sample_clean)
        y_final = train_df_sample['label'].values
        
        runner.record_data_info({
            **runner.training_summary.get('data', {}),
            'feature_shape': str(X_final.shape)
        })
        
        logger.info(f"特征维度: {X_final.shape}")
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_final, y_final, test_size=0.2, random_state=SEED, stratify=y_final
        )
        
        callbacks_final = create_lgbm_callbacks(logger, log_interval=100)
        early_stopping_final = create_early_stopping_callback(logger, patience=100)
        
        trainer_final, _ = train_and_evaluate(
            X_tr, y_tr, X_val, y_val,
            model_type='lgb',
            callbacks=callbacks_final + [early_stopping_final]
        )
        
        logger.info("模型训练完成")
    
    with tracker.stage('模型保存'):
        logger.info("保存模型和特征管道...")
        fe_final.save()
        trainer_final.save()
        
        model_size = 0
        for f in ['lgb_model.pkl', 'tfidf.pkl', 'onehot_encoders.pkl', 
                  'multi_hot_encoder.pkl', 'scaler.pkl', 'ip_extractor.pkl']:
            fpath = os.path.join(MODEL_DIR, f)
            if os.path.exists(fpath):
                model_size += os.path.getsize(fpath)
        
        runner.record_model_info({
            'type': 'lightgbm',
            'class_weights': CLASS_WEIGHTS,
            'model_size_mb': model_size / (1024 * 1024),
            'feature_shape': str(X_final.shape)
        })
        
        logger.info(f"模型文件大小: {model_size / (1024 * 1024):.2f} MB")
    
    with tracker.stage('预测推理'):
        logger.info("在验证集上进行预测...")
        valid_df = pd.read_parquet(VALID_FILE)
        result_df = predict_on_test_data(valid_df, fe_final, trainer_final)
        
        infer_time_ms = tracker.get_stage_duration('预测推理') or 0
        runner.record_inference_info({
            'samples': len(result_df),
            'prediction_distribution': result_df['pred_label'].value_counts().to_dict(),
            'time_ms': infer_time_ms
        })
    
    return avg_metrics

if __name__ == '__main__':
    import os
    
    runner = create_runner(log_level='INFO')
    
    try:
        metrics = runner.run(run_training_pipeline)
    except Exception as e:
        runner.logger.critical(f"训练流程失败: {e}")
        raise
