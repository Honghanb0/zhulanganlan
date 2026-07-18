import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score, roc_auc_score
)
import lightgbm as lgb
from config import MODEL_DIR, CLASS_WEIGHTS, SEED

class ModelTrainer:
    def __init__(self, model_type='lgb'):
        self.model_type = model_type
        self.model = None
    
    def train_lgb(self, X_train, y_train, X_val=None, y_val=None, params=None, callbacks=None):
        if params is None:
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'boosting_type': 'gbdt',
                'num_leaves': 256,
                'learning_rate': 0.03,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': SEED,
                'is_unbalance': True,
                'min_data_in_leaf': 30,
                'max_depth': -1,
                'reg_alpha': 0.05,
                'reg_lambda': 0.05,
                'n_estimators': 1000,
                'class_weight': CLASS_WEIGHTS
            }
        
        self.model = lgb.LGBMClassifier(**params)
        
        if X_val is not None and y_val is not None:
            if callbacks is None:
                callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=callbacks
            )
        else:
            self.model.fit(X_train, y_train, callbacks=callbacks)
        
        return self
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, params=None, callbacks=None):
        if self.model_type == 'lgb':
            self.train_lgb(X_train, y_train, X_val, y_val, params, callbacks)
        return self
    
    def predict(self, X):
        if self.model_type == 'lgb':
            return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model_type == 'lgb':
            return self.model.predict_proba(X)
    
    def save(self, path=None):
        if path is None:
            path = MODEL_DIR
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f'{self.model_type}_model.pkl')
        joblib.dump(self.model, model_path)
    
    @classmethod
    def load(cls, model_type='lgb', path=None):
        if path is None:
            path = MODEL_DIR
        trainer = cls(model_type=model_type)
        model_path = os.path.join(path, f'{model_type}_model.pkl')
        trainer.model = joblib.load(model_path)
        return trainer

def evaluate_model(y_true, y_pred, y_proba=None):
    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['benign', 'suspicious', 'malicious']))
    
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm, index=['benign', 'suspicious', 'malicious'], 
                       columns=['benign', 'suspicious', 'malicious']))
    
    malicious_recall = recall_score(y_true, y_pred, labels=[2], average=None)[0]
    suspicious_recall = recall_score(y_true, y_pred, labels=[1], average=None)[0]
    print(f"\nMalicious Recall: {malicious_recall:.4f}")
    print(f"Suspicious Recall: {suspicious_recall:.4f}")
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            print(f"AUC: {auc:.4f}")
        except:
            print("AUC calculation skipped")
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'malicious_recall': malicious_recall,
        'suspicious_recall': suspicious_recall
    }

def train_and_evaluate(X_train, y_train, X_val, y_val, model_type='lgb', use_optuna=False, callbacks=None):
    if use_optuna:
        try:
            import optuna
            from optuna.integration import LightGBMPruningCallback
            
            def objective(trial):
                params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                    'num_leaves': trial.suggest_int('num_leaves', 32, 512, step=32),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'random_state': SEED,
                    'is_unbalance': True,
                    'class_weight': CLASS_WEIGHTS
                }
                
                model = lgb.LGBMClassifier(**params, n_estimators=1000)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='multi_logloss',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        LightGBMPruningCallback(trial, 'multi_logloss')
                    ]
                )
                
                y_pred = model.predict(X_val)
                return f1_score(y_val, y_pred, average='macro')
            
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            
            print(f"\nOptuna best parameters: {study.best_params}")
            print(f"Optuna best score: {study.best_value:.4f}")
            
            trainer = ModelTrainer(model_type=model_type)
            best_params = study.best_params
            best_params['n_estimators'] = 1000
            best_params['class_weight'] = CLASS_WEIGHTS
            trainer.fit(X_train, y_train, X_val, y_val, params=best_params, callbacks=callbacks)
        except ImportError:
            print("Optuna not installed, using default parameters")
            trainer = ModelTrainer(model_type=model_type)
            trainer.fit(X_train, y_train, X_val, y_val, callbacks=callbacks)
    else:
        trainer = ModelTrainer(model_type=model_type)
        trainer.fit(X_train, y_train, X_val, y_val, callbacks=callbacks)
    
    y_pred = trainer.predict(X_val)
    y_proba = trainer.predict_proba(X_val)
    
    metrics = evaluate_model(y_val, y_pred, y_proba)
    
    trainer.save()
    return trainer, metrics