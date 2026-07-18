import pandas as pd
import numpy as np
from config import OUTPUT_FILE, REVERSE_LABEL_MAPPING
from data_loader import load_valid_data, preprocess_data
from pipeline import FeaturePipeline
from model_trainer import ModelTrainer

LEAKAGE_COLS = [
    'label_binary', 'lifecycle_stage', 'mo_name', 'matched_rules', 
    'set_roles', 'attack_techniques', 'attack_tactics', 'event_id',
    'severity', 'action', 'product_name', 'vendor_name', 'vendor_code',
    'username', 'src_ip', 'dst_ip', 'src_host', 'dst_host',
    'stream_name', 'message_type', 'message_sanitized',
    'label_confidence', 'suspicion_score', 'pipeline'
]

def predict_on_test_data(test_df, feature_pipeline, trainer, output_file=None):
    if output_file is None:
        output_file = OUTPUT_FILE
    
    test_df = preprocess_data(test_df, is_train=False)
    
    test_df_clean = test_df.drop(columns=[c for c in LEAKAGE_COLS if c in test_df.columns])
    X_test = feature_pipeline.transform(test_df_clean)
    
    y_pred = trainer.predict(X_test)
    y_pred_labels = [REVERSE_LABEL_MAPPING[int(pred)] for pred in y_pred]
    
    result_df = pd.DataFrame({
        'event_id': test_df['event_id'],
        'pred_label': y_pred_labels
    })
    
    result_df.to_csv(output_file, index=False)
    print(f"Prediction results saved to {output_file}")
    print(f"Prediction distribution:")
    print(result_df['pred_label'].value_counts())
    
    return result_df

def load_and_predict(test_file=None):
    from config import VALID_FILE
    
    if test_file is None:
        test_file = VALID_FILE
    
    print("Loading feature pipeline...")
    fe = FeaturePipeline.load()
    
    print("Loading model...")
    trainer = ModelTrainer.load(model_type='lgb')
    
    print("Loading test data...")
    test_df = load_valid_data()
    
    print("Making predictions...")
    return predict_on_test_data(test_df, fe, trainer)

if __name__ == '__main__':
    load_and_predict()