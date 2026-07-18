import logging
import os
import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

class TrainingLogger:
    def __init__(self, name: str, log_level: str = 'INFO', task_id: Optional[str] = None):
        self.name = name
        self.log_level = log_level
        self.task_id = task_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.propagate = False
        self._setup_handlers()
    
    def _setup_handlers(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        class CustomFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                dt = datetime.datetime.fromtimestamp(record.created)
                if datefmt:
                    return dt.strftime(datefmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        formatter = CustomFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        log_file = os.path.join(LOG_DIR, f'train_{self.task_id}.log')
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def get_log_file_path(self) -> str:
        for handler in self.logger.handlers:
            if isinstance(handler, TimedRotatingFileHandler):
                return handler.baseFilename
        return ''

_loggers = {}

def get_logger(name: str = 'training', log_level: str = 'INFO', task_id: Optional[str] = None) -> TrainingLogger:
    key = f"{name}_{task_id}" if task_id else name
    if key not in _loggers:
        _loggers[key] = TrainingLogger(name, log_level, task_id)
    return _loggers[key]
