import logging
from logging.handlers import TimedRotatingFileHandler
import sys,os
from threading import get_native_id

class NativeThreadFormatter(logging.Formatter):
    def format(self, record):
        # 添加原生线程ID到日志记录
        record.native_thread_id = get_native_id()
        return super().format(record)


# 日志格式
log_formatter = NativeThreadFormatter(
    fmt = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(native_thread_id)d][%(module)s:%(funcName)s:%(lineno)d] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

log_path = os.environ.get('LOG_PATH', 'log/service.log')
log_days = int(os.environ.get('LOG_DAYS', 30))
log_debug = os.environ.get('LOG_DEBUG', '0') == '1'
log_level = logging.DEBUG if log_debug else logging.INFO
os.makedirs(os.path.dirname(log_path), mode=0o755, exist_ok=True)

# 文件日志处理器 - 按天切分，保留最近30天日志
file_handler = TimedRotatingFileHandler(log_path, when='midnight', interval=1, backupCount=log_days, encoding='utf-8')
file_handler.suffix = "%Y%m%d"
file_handler.setFormatter(log_formatter)
file_handler.setLevel(log_level)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=log_level, handlers=[file_handler, console_handler])

logging.info(f"init logger config done, path:{log_path}, level_debug:{log_debug}, days:{log_days}")