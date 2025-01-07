from loguru import logger

from .config import Config


def verify(config: Config):
    logger.info(config)
