import time
import datetime
import traceback
import os
from functools import wraps
from typing import Dict, Any, Optional, Callable

from training_logger import get_logger
from resource_monitor import ResourceMonitor
from training_tracker import TrainingTracker
from lgbm_callbacks import create_lgbm_callbacks, create_early_stopping_callback
from report_generator import ReportGenerator

def retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = kwargs.get('logger') or get_logger('retry')
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {e}, 将在 {wait_time:.1f}s 后重试")
                    time.sleep(wait_time)
            
            logger.error(f"尝试 {max_retries} 次后仍失败，抛出异常: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator

class TrainingRunner:
    def __init__(self, log_level: str = 'INFO', task_id: Optional[str] = None):
        self.task_id = task_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = get_logger('training', log_level, self.task_id)
        self.monitor = ResourceMonitor(self.logger, sampling_interval=5)
        self.tracker = TrainingTracker(self.logger, self.monitor)
        self.report_generator = ReportGenerator(self.logger)
        self.training_summary: Dict[str, Any] = {}
        self._has_error = False
    
    def _format_duration(self, duration_ms: float) -> str:
        hours = int(duration_ms // 3600000)
        minutes = int((duration_ms % 3600000) // 60000)
        seconds = int((duration_ms % 60000) // 1000)
        milliseconds = int(duration_ms % 1000)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        elif minutes > 0:
            return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        else:
            return f"{seconds:02d}.{milliseconds:03d}"
    
    def _handle_exception(self, exc_type, exc_val, exc_tb):
        self._has_error = True
        
        error_msg = f"训练过程发生未处理异常: {exc_type.__name__}: {exc_val}"
        self.logger.critical(error_msg)
        
        stack_trace = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
        self.logger.critical(f"堆栈跟踪:\n{stack_trace}")
        
        self.training_summary['error'] = {
            'type': exc_type.__name__,
            'message': str(exc_val),
            'stack_trace': stack_trace
        }
    
    def run(self, training_func: Callable, *args, **kwargs):
        self.logger.info("=" * 60)
        self.logger.info("ZJS杯 AI竞赛 - 安全事件智能检测系统")
        self.logger.info(f"任务ID: {self.task_id}")
        self.logger.info("=" * 60)
        
        try:
            self.monitor.start()
            
            with self.tracker.stage('训练流程'):
                result = training_func(self, *args, **kwargs)
            
            self._generate_final_report()
            
            self.logger.info("=" * 60)
            self.logger.info("训练流程完成!")
            self.logger.info("=" * 60)
            
            return result
            
        except Exception as exc:
            self._handle_exception(type(exc), exc, exc.__traceback__)
            self._generate_final_report()
            raise
        
        finally:
            self.monitor.stop()
            self.monitor.log_summary()
            self.tracker.log_timing_summary()
    
    def _generate_final_report(self):
        try:
            self.training_summary.update({
                'task_id': self.task_id,
                'start_time': self.tracker.training_start_datetime.isoformat(),
                'end_time': datetime.datetime.now().isoformat(),
                'duration_ms': self.tracker.get_total_duration(),
                'duration_human': self._format_duration(self.tracker.get_total_duration()),
                'stages': self.tracker.get_stages_summary(),
                'resources': self.monitor.get_summary(),
                'has_error': self._has_error,
                'metrics': self.tracker.metrics
            })
            
            self.report_generator.generate_report(self.training_summary)
            
        except Exception as e:
            self.logger.error(f"生成报告时发生错误: {e}")
            traceback.print_exc()
    
    def record_data_info(self, data_info: Dict[str, Any]):
        self.training_summary['data'] = data_info
    
    def record_model_info(self, model_info: Dict[str, Any]):
        self.training_summary['model'] = model_info
    
    def record_cv_results(self, cv_results: Dict[str, Any]):
        self.training_summary['cross_validation'] = cv_results
    
    def record_inference_info(self, inference_info: Dict[str, Any]):
        self.training_summary['inference'] = inference_info
    
    def record_metrics_history(self, metrics_history: Dict[str, Any]):
        self.training_summary['metrics_history'] = metrics_history
    
    def record_feature_importance(self, feature_importance: Dict[str, float]):
        self.training_summary['feature_importance'] = feature_importance
    
    def get_log_file_path(self) -> str:
        return self.logger.get_log_file_path()

def create_runner(log_level: str = 'INFO', task_id: Optional[str] = None) -> TrainingRunner:
    return TrainingRunner(log_level, task_id)
