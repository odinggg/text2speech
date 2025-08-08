# src/text2speech/__init__.py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("requests").setLevel(logging.DEBUG) # 将requests日志设为WARNING，避免过多无关信息
logging.getLogger("urllib3").setLevel(logging.DEBUG)

from .main import TextToSpeechOrchestrator
from . import config

logger = logging.getLogger(__name__)

def main() -> None:
    try:
        orchestrator = TextToSpeechOrchestrator()
        metadata_directory = config.METADATA_DIR
        orchestrator.run_from_directory(metadata_directory)
    except KeyboardInterrupt:
        logger.info("User interrupted the program.")
    except Exception as e:
        logger.critical(f"A fatal error occurred during application startup: {e}", exc_info=True)
