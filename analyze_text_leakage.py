import pandas as pd
import numpy as np
from data_loader import load_train_data, preprocess_data

def analyze_text_patterns():
    print("加载数据...")
    train_df = load_train_data()
    train_df = preprocess_data(train_df, is_train=True)
    
    LEAKAGE_COLS = [
        'label_binary', 'lifecycle_stage', 'mo_name', 'matched_rules', 
        'set_roles', 'attack_techniques', 'attack_tactics', 'event_id',
        'severity', 'action', 'product_name', 'vendor_name', 'vendor_code',
        'username', 'src_ip', 'dst_ip', 'src_host', 'dst_host',
        'stream_name', 'message_type'
    ]
    
    df_clean = train_df.drop(columns=[c for c in LEAKAGE_COLS if c in train_df.columns])
    
    print(f"\n剩余特征: {df_clean.columns.tolist()}")
    
    print("\n分析message_sanitized与标签的关系...")
    
    label_text_counts = train_df.groupby('label')['message_sanitized'].nunique()
    print(f"每个标签的唯一消息数:")
    print(label_text_counts)
    
    text_label_groups = train_df.groupby('message_sanitized')['label'].nunique()
    single_label_texts = text_label_groups[text_label_groups == 1].count()
    multi_label_texts = text_label_groups[text_label_groups > 1].count()
    print(f"\n仅对应单个标签的文本数: {single_label_texts}")
    print(f"对应多个标签的文本数: {multi_label_texts}")
    print(f"单标签文本占比: {single_label_texts / (single_label_texts + multi_label_texts) * 100:.2f}%")
    
    text_to_label = train_df.groupby('message_sanitized')['label'].agg(lambda x: x.unique()).to_dict()
    label_dist = {}
    for text, labels in text_to_label.items():
        if len(labels) == 1:
            label = labels[0]
            if label not in label_dist:
                label_dist[label] = {'count': 0, 'examples': []}
            label_dist[label]['count'] += 1
            if len(label_dist[label]['examples']) < 5:
                label_dist[label]['examples'].append(text[:100])
    
    print("\n每个标签的单标签文本统计:")
    for label, data in label_dist.items():
        print(f"\n标签 {label}: {data['count']} 个唯一文本")
        print("示例:")
        for example in data['examples']:
            print(f"  {example}")

if __name__ == '__main__':
    analyze_text_patterns()
