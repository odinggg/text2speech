# src/text_to_speech_app/main.py
import asyncio
import os
import logging
from typing import Optional

import config
from .processor import TextToSpeechProcessor, TTSConfig, EmbeddingConfig

# 配置日志
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class TextToSpeechOrchestrator:
    """
    编排整个文本转语音流程。
    负责扫描任务目录，并为每个任务调用处理器。
    """
    def __init__(self, tts_config: Optional[TTSConfig] = None, embedding_config: Optional[EmbeddingConfig] = None):
        """
        初始化编排器。

        Args:
            tts_config (Optional[TTSConfig]): 一个可选的TTS配置对象。
            embedding_config (Optional[EmbeddingConfig]): 一个可选的Embedding配置对象。
        """
        self.processor = TextToSpeechProcessor(tts_config, embedding_config)
        logger.info("TextToSpeechOrchestrator initialized.")
        os.makedirs(config.VOICEDATA_DIR, exist_ok=True)

    async def run_from_directory(self, directory_path: str):
        """
        扫描指定目录下的元数据文件，并为每个待处理的任务启动音频生成流程。

        Args:
            directory_path (str): 包含元数据文件 (*-metadata.json) 的目录路径。
        """
        logger.info(f"Starting scan in directory: '{directory_path}'")

        if not os.path.isdir(directory_path):
            logger.error(f"Provided directory does not exist: {directory_path}")
            return

        tasks_to_process = []
        for filename in os.listdir(directory_path):
            if filename.endswith("-metadata.json"):
                tasks_to_process.append(os.path.join(directory_path, filename))

        if not tasks_to_process:
            logger.info("No metadata files found to process in the specified directory.")
            return

        logger.info(f"Found {len(tasks_to_process)} potential tasks.")

        for task_meta_path in tasks_to_process:
            await self.processor.run(metadata_path=task_meta_path)

        logger.info("All tasks in the directory have been processed.")
