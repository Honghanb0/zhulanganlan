import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import os

from data_loader import load_train_data, preprocess_data
from config import SEED

def audit_duplicates(df):
    print("=" * 60)
    print("1. 重复样本检查")
    print("=" * 60)
    
    exact_duplicates = df.duplicated().sum()
    print(f"精确重复样本数: {exact_duplicates}")
    
    if exact_duplicates > 0:
        duplicate_rows = df[df.duplicated(keep=False)]
        print(f"重复样本占比: {exact_duplicates / len(df) * 100:.2f}%")
        print("重复样本示例:")
        print(duplicate_rows.head(5))
    
    text_cols = ['message', 'message_sanitized'] if 'message' in df.columns else ['message_sanitized']
    for col in text_cols:
        if col in df.columns:
            text_duplicates = df[col].duplicated().sum()
            print(f"\n{col} 字段重复数: {text_duplicates}")
            print(f"{col} 字段重复占比: {text_duplicates / len(df) * 100:.2f}%")
    
    return exact_duplicates

def audit_label_correlation(df):
    print("\n" + "=" * 60)
    print("2. 特征与标签相关性分析")
    print("=" * 60)
    
    label_col = 'label' if 'label' in df.columns else 'label_binary'
    print(f"使用标签列: {label_col}")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for col in categorical_cols:
        if col == label_col:
            continue
        
        contingency = pd.crosstab(df[col], df[label_col], normalize='index')
        max_corr = contingency.max(axis=1).mean()
        
        unique_ratio = df[col].nunique() / len(df)
        
        if max_corr > 0.9:
            print(f"\n⚠️ 高相关性特征: {col}")
            print(f"  平均最大类别对应率: {max_corr:.4f}")
            print(f"  唯一值比率: {unique_ratio:.4f}")
            print(f"  列联表:")
            print(contingency)
    
    if numeric_cols and label_col in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df[label_col])
        
        X_numeric = df[numeric_cols].fillna(0).values
        
        try:
            mi_scores = mutual_info_classif(X_numeric, y, random_state=SEED)
            mi_df = pd.DataFrame({
                'feature': numeric_cols,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False)
            
            print("\n数值特征互信息得分:")
            print(mi_df.to_string(index=False))
            
            high_mi = mi_df[mi_df['mutual_info'] > 0.5]
            if not high_mi.empty:
                print(f"\n⚠️ 高互信息特征(MI > 0.5):")
                print(high_mi.to_string(index=False))
        except Exception as e:
            print(f"互信息计算失败: {e}")

def audit_text_label_leakage(df):
    print("\n" + "=" * 60)
    print("3. 文本字段标签词泄漏检查")
    print("=" * 60)
    
    text_cols = []
    for col in df.columns:
        if 'message' in col.lower() or 'text' in col.lower():
            text_cols.append(col)
    
    label_keywords = ['benign', 'malicious', 'suspicious', 'attack', 'threat', 'safe', 'normal', 'anomaly']
    
    for col in text_cols:
        print(f"\n检查字段: {col}")
        
        for keyword in label_keywords:
            count = df[col].astype(str).str.lower().str.contains(keyword).sum()
            if count > 0:
                ratio = count / len(df) * 100
                print(f"  '{keyword}' 出现次数: {count} ({ratio:.2f}%)")
                
                samples = df[df[col].astype(str).str.lower().str.contains(keyword)][[col, 'label_binary']].head(3)
                print("  示例:")
                for _, row in samples.iterrows():
                    print(f"    Label: {row['label_binary']}, Text: {str(row[col])[:100]}...")

def audit_identifier_label_mapping(df):
    print("\n" + "=" * 60)
    print("4. 标识符字段标签映射检查")
    print("=" * 60)
    
    id_cols = []
    for col in df.columns:
        if 'id' in col.lower() or 'session' in col.lower() or 'alert' in col.lower() or 'event' in col.lower():
            id_cols.append(col)
    
    label_col = 'label' if 'label' in df.columns else 'label_binary'
    
    for col in id_cols:
        print(f"\n检查字段: {col}")
        
        id_label_counts = df.groupby(col)[label_col].nunique()
        multiple_labels = (id_label_counts > 1).sum()
        single_label = (id_label_counts == 1).sum()
        
        print(f"  唯一值数量: {df[col].nunique()}")
        print(f"  仅映射到单个标签的ID数: {single_label}")
        print(f"  映射到多个标签的ID数: {multiple_labels}")
        
        if multiple_labels > 0:
            print(f"  多标签映射率: {multiple_labels / df[col].nunique() * 100:.2f}%")

def audit_categorical_label_leakage(df):
    print("\n" + "=" * 60)
    print("5. 分类特征标签泄漏检查")
    print("=" * 60)
    
    label_col = 'label' if 'label' in df.columns else 'label_binary'
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        if col == label_col or col == 'label_binary':
            continue
        
        total_samples = len(df)
        grouped = df.groupby(col)[label_col].value_counts(normalize=True).unstack(fill_value=0)
        
        if len(grouped.columns) >= 2:
            max_ratio = grouped.max(axis=1).mean()
            
            if max_ratio > 0.95:
                print(f"\n⚠️ 潜在泄漏特征: {col}")
                print(f"  平均类别纯度: {max_ratio:.4f}")
                
                pure_categories = grouped[grouped.max(axis=1) > 0.95].index.tolist()
                print(f"  高纯度类别数: {len(pure_categories)}")
                if len(pure_categories) < 10:
                    print(f"  高纯度类别: {pure_categories}")

def audit_data():
    print("=" * 80)
    print("数据质量审计报告")
    print("=" * 80)
    
    print("\n加载数据...")
    train_df = load_train_data()
    print(f"原始数据形状: {train_df.shape}")
    
    print("\n预处理数据...")
    train_df = preprocess_data(train_df, is_train=True)
    print(f"预处理后数据形状: {train_df.shape}")
    
    audit_duplicates(train_df)
    audit_label_correlation(train_df)
    audit_text_label_leakage(train_df)
    audit_identifier_label_mapping(train_df)
    audit_categorical_label_leakage(train_df)
    
    print("\n" + "=" * 80)
    print("数据审计完成")
    print("=" * 80)

if __name__ == '__main__':
    audit_data()
