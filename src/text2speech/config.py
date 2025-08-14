# src/text2speech/config.py
import os
from dotenv import load_dotenv

# [新增] 从 .env 文件加载环境变量，这对于本地开发非常方便
# 在生产环境中，您应该直接设置系统的环境变量
load_dotenv()

# --- 基础路径配置 (保持不变) ---
BASE_DIR = os.getenv("BASE_DIR", "./")
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METADATA_DIR = os.path.join(RESULTS_DIR, 'metadata')
SPLITDATA_DIR = os.path.join(RESULTS_DIR, 'splitdata')
VOICEDATA_DIR = os.path.join(RESULTS_DIR, 'voicedata')
VECTORTEMP_DIR = os.path.join(RESULTS_DIR, 'vectortemp')

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000)) # 每个块的目标最大字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))  # 细分时，块之间的重叠字符数
# --- TTS 服务配置 (从环境变量加载) ---
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8080:443")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "/tts/fast")
# 注意：环境变量返回的是字符串，需要转换为整数
TTS_REQUEST_TIMEOUT = int(os.getenv("TTS_REQUEST_TIMEOUT", 60000))

# --- Embedding 服务配置 (从环境变量加载) ---
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "123456")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://127.0.0.1:8080/v1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "qwen3")
# 注意：环境变量返回的是字符串，需要转换为浮点数
SEMANTIC_SPLIT_THRESHOLD = float(os.getenv("SEMANTIC_SPLIT_THRESHOLD", 0.3))

# --- 日志配置 (从环境变量加载) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

