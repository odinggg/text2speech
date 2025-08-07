# run.py
import asyncio
import logging

# 从我们创建的包中导入主编排器类和默认配置
from src.text_to_speech_app.main import TextToSpeechOrchestrator
from src.text_to_speech_app import config

# 获取日志记录器
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    """
    应用执行入口。
    使用 `python run.py` 来启动。
    """
    try:
        # 1. 创建编排器实例
        orchestrator = TextToSpeechOrchestrator()

        # 2. 指定要处理的元数据目录并运行
        #    默认指向 results/metadata/
        metadata_directory = config.METADATA_DIR

        # 3. 使用 asyncio.run 来执行异步的 run_from_directory 方法
        asyncio.run(orchestrator.run_from_directory(metadata_directory))

    except KeyboardInterrupt:
        logger.info("User interrupted the program.")
    except Exception as e:
        # 捕获其他任何未预料到的错误
        logger.critical(f"A fatal error occurred during application startup: {e}", exc_info=True)

