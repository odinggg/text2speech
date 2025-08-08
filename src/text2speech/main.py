# src/text2speech/main.py
import os
import logging
from typing import Optional

from . import config
from .processor import TextToSpeechProcessor, TTSConfig, EmbeddingConfig

logger = logging.getLogger(__name__)

class TextToSpeechOrchestrator:
    def __init__(self, tts_config: Optional[TTSConfig] = None, embedding_config: Optional[EmbeddingConfig] = None):
        self.processor = TextToSpeechProcessor(tts_config, embedding_config)
        logger.info("TextToSpeechOrchestrator initialized.")
        os.makedirs(config.VOICEDATA_DIR, exist_ok=True)

    def run_from_directory(self, directory_path: str):
        logger.info(f"Starting scan in directory: '{directory_path}'")
        if not os.path.isdir(directory_path):
            logger.error(f"Provided directory does not exist: {directory_path}")
            return

        tasks_to_process = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith("-metadata.json")]
        if not tasks_to_process:
            logger.info("No metadata files found to process.")
            return

        logger.info(f"Found {len(tasks_to_process)} potential tasks.")
        for task_meta_path in tasks_to_process:
            self.processor.run(metadata_path=task_meta_path)
        logger.info("All tasks in the directory have been processed.")
