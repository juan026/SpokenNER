# File: Logger.py
# Created by Moncef Benaicha
# Contact: support@moncefbenaicha.me


"""Logging configuration module"""
from logging.config import dictConfig
import logging
from logging import FileHandler


class Logger(object):
    """
    Logger class is Factory for logging service, that will init the logging formatting, also logging streams
    Every logging created by this class will have a console stream log
    """

    def __init__(self, level: str = "INFO"):
        """
        @param path: Path where to save log file
        """

        self.min_level = level
        self.config = {
            "version": 1,
            "disable_existing_loggers": True if level == "INFO" else False,
        }
        self.__add_formatter()
        self.config["root"] = {
            "level": level,
            "handlers": self.__add_handlers(level=level),
        }
        dictConfig(self.config)

    def __add_formatter(self):
        formatter = {
            "default": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s] : %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        }
        self.config["formatters"] = formatter

    def __add_handlers(self, level):
        handler = {
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            }
        }
        self.config["handlers"] = handler
        return ["consoleHandler"]

    def get_logger(self, name="Application Run", level="INFO", path=None):
        """
            get_logger function returns a logger object from logging package
            In addition to console logging stream, this function will add another log stream to a file,
            if not given the file will be by default ./application.log
        :param name: Log Name by default Application Run
        :param level: logging level by default INFO
        :param path: File Handler logging stream destination path
        :return: logger
        """
        logger = logging.getLogger(name or __name__)
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt=self.config["formatters"]["default"]["format"],
            datefmt=self.config["formatters"]["default"]["datefmt"],
        )
        if not path:
            path = "./application.log"
        handler = FileHandler(filename=path, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def change_log_output_file(self, logger, path):
        formatter = logging.Formatter(
            fmt=self.config["formatters"]["default"]["format"],
            datefmt=self.config["formatters"]["default"]["datefmt"],
        )
        handler = FileHandler(filename=path, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.handlers[0] = handler
