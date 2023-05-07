import logging.config
from config import CONFIG

# 配置日志文件和日志级别
# CRITICAL: 50
# ERROR: 40
# WARNING: 30
# INFO: 20
# DEBUG: 10
# NOTSET: 0

config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s~%(levelname)s:%(levelno)s~%(filename)s:%(lineno)d~%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'ERROR',
            'formatter': 'simple'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': CONFIG['error_log_path'],
            'level': 'DEBUG',
            'formatter': 'simple',
            'mode': 'w',
            'maxBytes': 10 ** 7,
            'backupCount': 3,

        },
        'debug_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': CONFIG['debug_log_path'],
            'level': 'DEBUG',
            'formatter': 'simple',
            'mode': 'w',
            'maxBytes': 10 ** 7,
            'backupCount': 3,

        },
        'operation_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': CONFIG['operation_log_path'],
            'level': 'DEBUG',
            'formatter': 'simple',
            'mode': 'w',
            'maxBytes': 10 ** 7,
            'backupCount': 3,
        },
    },
    'loggers': {
        'error': {
            'handlers': ['console', 'error_file'],
            'level': 'ERROR',
        },
        'debug': {
            'handlers': ['debug_file'],
            'level': 'DEBUG',
        },
        'simple': {
            'handlers': ['console'],
            'level': 'WARN',
        },
        'operation': {
            'handlers': ['operation_file'],
            'level': 'DEBUG',
        }
    }
}


class HandleConfLog(object):
    def __init__(self):
        logging.config.dictConfig(config)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

    def error_log(self):
        self.logger = logging.getLogger('error')
        return self.logger

    def debug_log(self):
        self.logger = logging.getLogger('debug')
        return self.logger

    def operation_log(self):
        self.logger = logging.getLogger('operation')
        return self.logger


error_log = HandleConfLog().error_log()
debug_log = HandleConfLog().debug_log()
operation_log = HandleConfLog().operation_log()
if __name__=='__main__':
    error_log.error('config_error_level')
    debug_log.debug('config_debug_level')
