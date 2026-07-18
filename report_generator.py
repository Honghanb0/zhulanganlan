import os
import json
import platform
import sys
import time
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

class ReportGenerator:
    def __init__(self, logger):
        self.logger = logger
    
    def _get_environment_info(self) -> Dict[str, str]:
        try:
            import lightgbm as lgb
            import pandas as pd
            import numpy as np
            import sklearn
            import psutil
        except ImportError:
            pass
        
        return {
            'system': platform.system(),
            'system_version': platform.version(),
            'architecture': platform.architecture()[0],
            'python_version': sys.version.split()[0],
            'lightgbm_version': getattr(lgb, '__version__', 'unknown') if 'lgb' in dir() else 'unknown',
            'pandas_version': getattr(pd, '__version__', 'unknown') if 'pd' in dir() else 'unknown',
            'numpy_version': getattr(np, '__version__', 'unknown') if 'np' in dir() else 'unknown',
            'sklearn_version': getattr(sklearn, '__version__', 'unknown') if 'sklearn' in dir() else 'unknown',
            'psutil_version': getattr(psutil, '__version__', 'unknown') if 'psutil' in dir() else 'unknown',
            'cpu_count': str(os.cpu_count()),
            'memory_gb': str(round(psutil.virtual_memory().total / (1024 ** 3), 2)) if 'psutil' in dir() else 'unknown'
        }
    
    def _generate_training_curve_plot(self, metrics_history: Dict, output_path: str):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            if metrics_history.get('training'):
                for metric_name, values in metrics_history['training'].items():
                    axes[0].plot(values, label=f'train_{metric_name}')
            
            if metrics_history.get('validation'):
                for metric_name, values in metrics_history['validation'].items():
                    axes[1].plot(values, label=f'val_{metric_name}')
            
            axes[0].set_title('Training Metrics')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Metric Value')
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].set_title('Validation Metrics')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Metric Value')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            self.logger.warning(f"Failed to generate training curve plot: {e}")
            return False
    
    def _generate_feature_importance_plot(self, feature_importance: Dict, output_path: str, top_n: int = 20):
        try:
            if not feature_importance:
                return False
            
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            sorted_indices = np.argsort(importances)[::-1][:top_n]
            sorted_features = [features[i] for i in sorted_indices]
            sorted_importances = [importances[i] for i in sorted_indices]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(sorted_features)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.gca().invert_yaxis()
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            self.logger.warning(f"Failed to generate feature importance plot: {e}")
            return False
    
    def _generate_metrics_table(self, metrics: Dict) -> str:
        rows = []
        for key, value in metrics.items():
            if isinstance(value, float):
                rows.append(f"| {key} | {value:.4f} |")
            else:
                rows.append(f"| {key} | {value} |")
        return '\n'.join(rows)
    
    def generate_markdown_report(self, training_summary: Dict, output_file: str = None) -> str:
        if output_file is None:
            output_file = os.path.join(LOG_DIR, 'train_report.md')
        
        report_parts = []
        
        report_parts.append("# 训练报告")
        report_parts.append(f"*生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
        report_parts.append("")
        
        report_parts.append("## 1. 训练概览")
        report_parts.append("| 项目 | 值 |")
        report_parts.append("|------|-----|")
        report_parts.append(f"| 开始时间 | {training_summary.get('start_time', 'N/A')} |")
        report_parts.append(f"| 结束时间 | {training_summary.get('end_time', 'N/A')} |")
        report_parts.append(f"| 总耗时 | {training_summary.get('duration_human', 'N/A')} |")
        report_parts.append(f"| 随机种子 | {training_summary.get('seed', 'N/A')} |")
        report_parts.append(f"| 交叉验证折数 | {training_summary.get('num_folds', 'N/A')} |")
        report_parts.append("")
        
        report_parts.append("## 2. 数据统计")
        data_info = training_summary.get('data', {})
        report_parts.append("| 项目 | 值 |")
        report_parts.append("|------|-----|")
        report_parts.append(f"| 总样本数 | {data_info.get('total_samples', 'N/A')} |")
        report_parts.append(f"| 交叉验证样本数 | {data_info.get('cv_samples', 'N/A')} |")
        report_parts.append(f"| 特征维度 | {data_info.get('feature_shape', 'N/A')} |")
        if data_info.get('label_distribution'):
            for label, count in data_info['label_distribution'].items():
                report_parts.append(f"| 标签_{label} | {count} |")
        report_parts.append("")
        
        report_parts.append("## 3. 交叉验证结果")
        cv_info = training_summary.get('cross_validation', {})
        avg_metrics = cv_info.get('avg_metrics', {})
        
        report_parts.append("### 平均指标")
        report_parts.append("| 指标 | 值 |")
        report_parts.append("|------|-----|")
        for key, value in avg_metrics.items():
            if isinstance(value, float):
                report_parts.append(f"| {key} | {value:.4f} |")
        
        report_parts.append("")
        report_parts.append("### 各折指标")
        fold_metrics = cv_info.get('fold_metrics', [])
        for i, metrics in enumerate(fold_metrics):
            report_parts.append(f"#### Fold {i+1}")
            report_parts.append("| 指标 | 值 |")
            report_parts.append("|------|-----|")
            for key, value in metrics.items():
                if isinstance(value, float):
                    report_parts.append(f"| {key} | {value:.4f} |")
            report_parts.append("")
        
        report_parts.append("## 4. 资源使用")
        resource_info = training_summary.get('resources', {})
        report_parts.append("| 项目 | 值 |")
        report_parts.append("|------|-----|")
        report_parts.append(f"| 峰值内存 | {resource_info.get('peak_memory_mb', 0):.1f} MB |")
        report_parts.append(f"| 平均内存 | {resource_info.get('avg_memory_mb', 0):.1f} MB |")
        report_parts.append(f"| 平均CPU使用率 | {resource_info.get('avg_cpu_percent', 0):.1f}% |")
        report_parts.append(f"| 磁盘读取 | {resource_info.get('total_disk_io_read_mb', 0):.1f} MB |")
        report_parts.append(f"| 磁盘写入 | {resource_info.get('total_disk_io_write_mb', 0):.1f} MB |")
        report_parts.append("")
        
        report_parts.append("## 5. 训练阶段耗时")
        stages_info = training_summary.get('stages', {})
        report_parts.append("| 阶段 | 耗时 |")
        report_parts.append("|------|------|")
        for stage_name, duration_ms in stages_info.items():
            seconds = duration_ms / 1000
            report_parts.append(f"| {stage_name} | {seconds:.2f}s |")
        report_parts.append("")
        
        report_parts.append("## 6. 模型信息")
        model_info = training_summary.get('model', {})
        report_parts.append("| 项目 | 值 |")
        report_parts.append("|------|-----|")
        report_parts.append(f"| 模型类型 | {model_info.get('type', 'N/A')} |")
        report_parts.append(f"| 模型文件大小 | {model_info.get('model_size_mb', 0):.2f} MB |")
        report_parts.append("")
        
        if training_summary.get('training_curve_plot'):
            report_parts.append("## 7. 训练曲线")
            report_parts.append(f"![Training Curve]({os.path.basename(training_summary['training_curve_plot'])})")
            report_parts.append("")
        
        if training_summary.get('feature_importance_plot'):
            report_parts.append("## 8. 特征重要性")
            report_parts.append(f"![Feature Importance]({os.path.basename(training_summary['feature_importance_plot'])})")
            report_parts.append("")
        
        report_parts.append("## 9. 运行环境")
        env_info = training_summary.get('environment', {})
        report_parts.append("| 项目 | 值 |")
        report_parts.append("|------|-----|")
        for key, value in env_info.items():
            report_parts.append(f"| {key} | {value} |")
        
        report_content = '\n'.join(report_parts)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"训练报告已保存至: {output_file}")
        return output_file
    
    def generate_json_summary(self, training_summary: Dict, output_file: str = None) -> str:
        if output_file is None:
            output_file = os.path.join(LOG_DIR, 'training_summary.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"训练摘要已保存至: {output_file}")
        return output_file
    
    def generate_report(self, training_summary: Dict) -> Dict[str, str]:
        self.logger.info("开始生成训练报告...")
        
        training_summary['environment'] = self._get_environment_info()
        
        training_curve_plot = os.path.join(LOG_DIR, 'training_curve.png')
        if self._generate_training_curve_plot(
            training_summary.get('metrics_history', {}), 
            training_curve_plot
        ):
            training_summary['training_curve_plot'] = training_curve_plot
        
        feature_importance_plot = os.path.join(LOG_DIR, 'feature_importance.png')
        if self._generate_feature_importance_plot(
            training_summary.get('feature_importance', {}), 
            feature_importance_plot
        ):
            training_summary['feature_importance_plot'] = feature_importance_plot
        
        md_file = self.generate_markdown_report(training_summary)
        json_file = self.generate_json_summary(training_summary)
        
        self.logger.info("训练报告生成完成")
        
        return {
            'markdown_report': md_file,
            'json_summary': json_file,
            'training_curve_plot': training_curve_plot,
            'feature_importance_plot': feature_importance_plot
        }
