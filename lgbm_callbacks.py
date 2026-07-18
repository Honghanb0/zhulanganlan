import lightgbm as lgb
import time
from typing import Dict, Any, List

class LGBMTrainingLogger:
    def __init__(self, logger, log_interval: int = 50):
        self.logger = logger
        self.log_interval = log_interval
        self.iteration_times: List[float] = []
        self.training_metrics: Dict[str, List[float]] = {}
        self.validation_metrics: Dict[str, List[float]] = {}
    
    def __call__(self, env):
        iteration = env.iteration
        start_time = time.time()
        
        if iteration % self.log_interval == 0 or iteration == env.end_iteration - 1:
            log_msg = f"[LGBM] Iteration {iteration:4d}"
            
            for data_name, metric_name, value, _ in env.evaluation_result_list:
                metric_key = f"{data_name}_{metric_name}"
                if data_name == 'training':
                    if metric_key not in self.training_metrics:
                        self.training_metrics[metric_key] = []
                    self.training_metrics[metric_key].append(value)
                else:
                    if metric_key not in self.validation_metrics:
                        self.validation_metrics[metric_key] = []
                    self.validation_metrics[metric_key].append(value)
                
                log_msg += f" | {data_name}_{metric_name}: {value:.6f}"
            
            duration = time.time() - start_time
            self.iteration_times.append(duration)
            
            avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)
            log_msg += f" | time: {duration:.3f}s | avg: {avg_iter_time:.3f}s"
            
            self.logger.info(log_msg)
    
    def get_metrics_history(self) -> Dict[str, Any]:
        return {
            'training': self.training_metrics,
            'validation': self.validation_metrics,
            'iteration_times': self.iteration_times
        }

def create_lgbm_callbacks(logger, log_interval: int = 50) -> List:
    return [
        LGBMTrainingLogger(logger, log_interval)
    ]

def create_early_stopping_callback(logger, patience: int = 100):
    return lgb.early_stopping(
        stopping_rounds=patience,
        verbose=True
    )
