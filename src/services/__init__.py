import os
from .logger import Logger
import sys
from better_exceptions import format_exception
import requests
import transformers

log_factory = Logger()

logger = log_factory.get_logger()

task: str = 'UKN Task'

DESTINATION = os.environ.get("DESTINATION")
NOTIFICATION_SERVICE_TOKEN = os.environ.get("NOTIFICATION_SERVICE_TOKEN")
NOTIFICATION_SERVICE_URL = os.environ.get("NOTIFICATION_SERVICE_URL")


__notification_service = True if all([DESTINATION, NOTIFICATION_SERVICE_TOKEN, NOTIFICATION_SERVICE_URL]) else False

if __notification_service:
    logger.info(
        f"Notification service is active on: {NOTIFICATION_SERVICE_URL.format(NOTIFICATION_SERVICE_TOKEN, DESTINATION, 'MESSAGE')}")


def send_notification(message):
    if __notification_service:
        message = f'From: {task} -- {message}'
        url = NOTIFICATION_SERVICE_URL.format(NOTIFICATION_SERVICE_TOKEN, DESTINATION, message)
        _ = requests.get(url)


def handle_exception(exc, value, tb):
    message = f'Error due to: {type(value).__name__}: {value}'
    send_notification(message)
    logger.error(u''.join(format_exception(exc, value, tb)))


sys.excepthook = handle_exception
transformers.logger = logger
transformers.trainer.logger = logger