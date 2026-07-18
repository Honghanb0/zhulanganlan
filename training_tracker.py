import time
import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from training_logger import TrainingLogger
from resource_monitor import ResourceMonitor

@dataclass
class StageInfo:
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None

class StageTimer:
    def __init__(self, ctx, stage_name: str):
        self.ctx = ctx
        self.stage_name = stage_name
    
    def __enter__(self):
        self.ctx.start_stage(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.end_stage(self.stage_name)

class TrainingTracker:
    def __init__(self, logger: TrainingLogger, monitor: ResourceMonitor):
        self.logger = logger
        self.monitor = monitor
        self.stages: Dict[str, StageInfo] = {}
        self.current_stage: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.start_time = time.time()
        self.training_start_datetime = datetime.datetime.now()
    
    def start_stage(self, stage_name: str):
        if self.current_stage:
            self.logger.warning(f"Starting stage '{stage_name}' but '{self.current_stage}' is still active")
        
        self.current_stage = stage_name
        start_time = time.time()
        self.stages[stage_name] = StageInfo(
            name=stage_name,
            start_time=start_time
        )
        
        formatted_time = self._format_time(start_time)
        self.logger.info(f"[{stage_name}] - 开始执行")
    
    def end_stage(self, stage_name: str):
        if stage_name not in self.stages:
            self.logger.warning(f"Ending stage '{stage_name}' which was never started")
            return
        
        end_time = time.time()
        stage_info = self.stages[stage_name]
        stage_info.end_time = end_time
        stage_info.duration_ms = (end_time - stage_info.start_time) * 1000
        
        self.current_stage = None
        
        formatted_duration = self._format_duration(stage_info.duration_ms)
        self.logger.info(f"[{stage_name}] - 执行完成，耗时: {formatted_duration}")
    
    def stage(self, stage_name: str) -> StageTimer:
        return StageTimer(self, stage_name)
    
    def record_metrics(self, metrics: Dict[str, Any], prefix: str = ''):
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            self.metrics[full_key] = value
    
    def get_stage_duration(self, stage_name: str) -> Optional[float]:
        if stage_name in self.stages and self.stages[stage_name].duration_ms:
            return self.stages[stage_name].duration_ms
        return None
    
    def get_total_duration(self) -> float:
        return (time.time() - self.start_time) * 1000
    
    def get_stages_summary(self) -> Dict[str, float]:
        summary = {}
        for name, info in self.stages.items():
            if info.duration_ms:
                summary[name] = info.duration_ms
        return summary
    
    def _format_time(self, timestamp: float) -> str:
        return datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S.%f')[:-3]
    
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
    
    def log_timing_summary(self):
        self.logger.info("=" * 60)
        self.logger.info("训练阶段耗时统计")
        self.logger.info("=" * 60)
        
        total_duration = self.get_total_duration()
        
        for name, info in self.stages.items():
            if info.duration_ms:
                formatted_duration = self._format_duration(info.duration_ms)
                percentage = (info.duration_ms / total_duration) * 100 if total_duration > 0 else 0
                self.logger.info(f"  [{name}] : {formatted_duration} ({percentage:.1f}%)")
        
        formatted_total = self._format_duration(total_duration)
        self.logger.info(f"  [总耗时] : {formatted_total}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        return {
            'start_time': self.training_start_datetime.isoformat(),
            'end_time': datetime.datetime.now().isoformat(),
            'total_duration_ms': self.get_total_duration(),
            'stages': {name: info.duration_ms for name, info in self.stages.items() if info.duration_ms},
            'metrics': self.metrics
        }
