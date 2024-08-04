import os
import logging
import logging.config

def setup_logger(args, logdir):
    """Creates and returns a fancy logger."""
    # return logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    # Why is setting up proper logging so !@?#! ugly?
    os.makedirs(os.path.join(logdir, args.name), exist_ok=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "stderr": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "logfile": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": os.path.join(logdir, args.name, "train.log"),
                "mode": "a",
            }
        },
        "loggers": {
            "": {
                "handlers": ["stderr", "logfile"],
                "level": "DEBUG",
                "propagate": True
            },
        }
    })
    logger = logging.getLogger(__name__)
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    logger.info(args)
    return logger
