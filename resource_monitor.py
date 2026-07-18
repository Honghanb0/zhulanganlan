import psutil
import os
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ResourceSample:
    timestamp: float
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0

class ResourceMonitor:
    def __init__(self, logger, sampling_interval: int = 5):
        self.logger = logger
        self.sampling_interval = sampling_interval
        self.samples: List[ResourceSample] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._process = psutil.Process(os.getpid())
        self._disk_io_start = psutil.disk_io_counters()
    
    def _sample(self):
        try:
            memory_info = self._process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self._process.memory_percent()
            cpu_percent = self._process.cpu_percent()
            
            disk_io_current = psutil.disk_io_counters()
            disk_io_read_mb = (disk_io_current.read_bytes - self._disk_io_start.read_bytes) / (1024 * 1024)
            disk_io_write_mb = (disk_io_current.write_bytes - self._disk_io_start.write_bytes) / (1024 * 1024)
            
            sample = ResourceSample(
                timestamp=time.time(),
                memory_usage_mb=memory_usage_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb
            )
            self.samples.append(sample)
            
            if len(self.samples) % 12 == 0:
                self.logger.debug(f"Resource sample: Memory={memory_usage_mb:.1f}MB ({memory_percent:.1f}%), CPU={cpu_percent:.1f}%")
        except Exception as e:
            self.logger.warning(f"Resource sampling error: {e}")
    
    def _run(self):
        while self._running:
            self._sample()
            time.sleep(self.sampling_interval)
    
    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self.logger.info("Resource monitoring started")
    
    def stop(self):
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join(timeout=5)
            self._sample()
            self.logger.info("Resource monitoring stopped")
    
    def get_current_stats(self) -> Dict[str, float]:
        memory_info = self._process.memory_info()
        return {
            'memory_mb': memory_info.rss / (1024 * 1024),
            'memory_percent': self._process.memory_percent(),
            'cpu_percent': self._process.cpu_percent()
        }
    
    def get_summary(self) -> Dict[str, float]:
        if not self.samples:
            return {}
        
        memory_usage = [s.memory_usage_mb for s in self.samples]
        memory_percent = [s.memory_percent for s in self.samples]
        cpu_percent = [s.cpu_percent for s in self.samples]
        
        return {
            'peak_memory_mb': max(memory_usage),
            'avg_memory_mb': sum(memory_usage) / len(memory_usage),
            'peak_memory_percent': max(memory_percent),
            'avg_memory_percent': sum(memory_percent) / len(memory_percent),
            'avg_cpu_percent': sum(cpu_percent) / len(cpu_percent),
            'total_disk_io_read_mb': self.samples[-1].disk_io_read_mb,
            'total_disk_io_write_mb': self.samples[-1].disk_io_write_mb,
            'num_samples': len(self.samples),
            'monitoring_duration_sec': self.samples[-1].timestamp - self.samples[0].timestamp
        }
    
    def log_summary(self):
        summary = self.get_summary()
        if summary:
            self.logger.info(f"Resource usage summary:")
            self.logger.info(f"  Peak memory: {summary['peak_memory_mb']:.1f}MB ({summary['peak_memory_percent']:.1f}%)")
            self.logger.info(f"  Avg memory: {summary['avg_memory_mb']:.1f}MB ({summary['avg_memory_percent']:.1f}%)")
            self.logger.info(f"  Avg CPU: {summary['avg_cpu_percent']:.1f}%")
            self.logger.info(f"  Disk I/O read: {summary['total_disk_io_read_mb']:.1f}MB")
            self.logger.info(f"  Disk I/O write: {summary['total_disk_io_write_mb']:.1f}MB")
