import logging.config
import os


def setup_logging(level="DEBUG",
                  directory="log",
                  filename=None,
                  log_format=None,
                  log_config=None):
    """
    A convenient way to configure your logger.

    """

    if filename is None:
        filename = "log"
    if not filename.endswith(".log"):
        filename += ".log"

    filename = os.path.join(directory, filename)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    fmt = '%(asctime)s-[%(levelname)-8s]-%(name)-25s: %(message)s'
    if log_format is not None:
        fmt = log_format

    configured_log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': fmt
            },
        },
        'handlers': {
            'default': {
                'level': "DEBUG",
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': filename,
                'when': 'midnight',
                'formatter': 'standard',
                'encoding': 'utf-8'
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': level,
                'propagate': True
            }
        }
    }

    if log_config is not None:
        configured_log_config = log_config

    logging.config.dictConfig(configured_log_config)
