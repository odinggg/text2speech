# src/text_to_speech_app/config.py
import os

# --- 基础路径配置 ---
# 获取项目根目录 (text_to_speech_project)
# 由于此文件在 src/text_to_speech_app/ 下，我们需要向上移动三级
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METADATA_DIR = os.path.join(RESULTS_DIR, 'metadata')
SPLITDATA_DIR = os.path.join(RESULTS_DIR, 'splitdata')
VOICEDATA_DIR = os.path.join(RESULTS_DIR, 'voicedata')

# --- 文本分块配置 ---
# Langchain 文本分块器使用
# Chunk Size: 每个文本块的目标大小（字符数）
CHUNK_SIZE = 1000
# Chunk Overlap: 相邻文本块之间的重叠字符数，以保持上下文连续性
CHUNK_OVERLAP = 0

# --- TTS 服务配置 ---
# 您的 TTS 服务 API 地址
TTS_BASE_URL = "http://127.0.0.1:8000"
TTS_ENDPOINT = "/tts/fast"
# 请求超时时间（秒），以防止长时间等待
TTS_REQUEST_TIMEOUT = 600
# 并发请求数，以避免对TTS服务器造成过大压力
MAX_CONCURRENT_REQUESTS = 1

EMBEDDING_API_KEY = "YOUR_API_KEY_HERE"
EMBEDDING_BASE_URL = "https://opengpt.fuckll.com/v1"
# 常见的 Embedding 模型名称
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
# 语义分割阈值：当相邻句子的相似度低于此值时，进行分割。
# 值越低，分割出的块越大；值越高，块越小。推荐范围 0.2-0.5
SEMANTIC_SPLIT_THRESHOLD = 0.3

# --- 日志配置 ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
