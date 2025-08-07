# src/text_to_speech_app/processor.py
import asyncio
import os
import uuid
import logging
from datetime import datetime
import json
from typing import List, Dict, Any, TypedDict, Optional

import aiohttp
import aiofiles
from pydub import AudioSegment
from tqdm import tqdm
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import OpenAIEmbeddings # <-- 替换为 Embedding 客户端
import numpy as np
import nltk

import config
from .text_utils import clean_text_for_tts

logger = logging.getLogger(__name__)

# 尝试下载NLTK的句子分割模型，如果失败则提示用户
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    logger.info("'punkt' downloaded successfully.")

# --- Pydantic Models for Configuration ---
class TTSConfig(BaseModel):
    base_url: str = Field(default=config.TTS_BASE_URL)
    endpoint: str = Field(default=config.TTS_ENDPOINT)
    timeout: int = Field(default=config.TTS_REQUEST_TIMEOUT)
    max_concurrent_requests: int = Field(default=config.MAX_CONCURRENT_REQUESTS)

class EmbeddingConfig(BaseModel):
    """用于语义文本分块的 Embedding 服务配置。"""
    api_key: SecretStr = Field(default=config.EMBEDDING_API_KEY)
    base_url: Optional[str] = Field(default=config.EMBEDDING_BASE_URL)
    model_name: str = Field(default=config.EMBEDDING_MODEL_NAME)
    split_threshold: float = Field(default=config.SEMANTIC_SPLIT_THRESHOLD,
                                   description="语义分割阈值")

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    metadata_path: str
    task_id: str
    metadata: Dict[str, Any]
    task_voicedata_dir: str
    final_audio_paths: List[str]
    processing_start_time: str

class TextToSpeechProcessor:
    """使用 LangGraph 管理文本转语音的工作流。"""
    def __init__(self, tts_config: Optional[TTSConfig] = None, embedding_config: Optional[EmbeddingConfig] = None):
        """初始化处理器，配置TTS和Embedding客户端。"""
        self.tts_config = tts_config or TTSConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()

        try:
            self.embedding_client = OpenAIEmbeddings(
                model=self.embedding_config.model_name,
                api_key=self.embedding_config.api_key,
                base_url=self.embedding_config.base_url,
            )
            logger.info(f"Embedding 客户端初始化成功。模型: {self.embedding_config.model_name}")
        except Exception:
            logger.error("初始化 Embedding 客户端失败！请检查 API Key 或网络连接。", exc_info=True)
            raise

        self.app = self._build_graph()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量之间的余弦相似度。"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    async def _intelligent_split(self, text: str) -> List[str]:
        """使用 Embedding 和语义相似度将文本分割成块。"""
        logger.debug("正在使用 Embedding 进行语义文本分块...")
        # 1. 使用 NLTK 将文本分割成句子
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            logger.debug("文本只有一个句子或更少，无需分割。")
            return sentences

        try:
            # 2. 为所有句子批量生成 Embedding 向量
            logger.debug(f"正在为 {len(sentences)} 个句子生成 embeddings...")
            embeddings = await self.embedding_client.aembed_documents(sentences)
            embeddings = np.array(embeddings)
            logger.debug("Embeddings 生成完毕。")

            # 3. 计算相邻句子之间的余弦相似度
            similarities = [
                self._cosine_similarity(embeddings[i], embeddings[i+1])
                for i in range(len(embeddings) - 1)
            ]

            # 4. 根据阈值确定分割点
            chunks = []
            current_chunk_start_index = 0
            for i, similarity in enumerate(similarities):
                if similarity < self.embedding_config.split_threshold:
                    # 语义差异大，在此处分割
                    chunk_sentences = sentences[current_chunk_start_index : i + 1]
                    chunks.append(" ".join(chunk_sentences))
                    current_chunk_start_index = i + 1

            # 5. 添加最后一个块
            last_chunk_sentences = sentences[current_chunk_start_index:]
            if last_chunk_sentences:
                chunks.append(" ".join(last_chunk_sentences))

            logger.debug(f"文本被成功分割成 {len(chunks)} 个语义块。")
            return chunks

        except Exception:
            logger.error("语义分块时出错，将回退到将整个文本作为单个块返回。", exc_info=True)
            return [text]

    # ... _fetch_tts_audio 和 _merge_and_cleanup_chunks 方法保持不变 ...
    async def _fetch_tts_audio(self, session: aiohttp.ClientSession, text: str, output_path: str) -> bool:
        tts_url = f"{self.tts_config.base_url}{self.tts_config.endpoint}"
        form_data = aiohttp.FormData()
        form_data.add_field('text', text)
        try:
            timeout = aiohttp.ClientTimeout(total=self.tts_config.timeout)
            async with session.post(tts_url, data=form_data, timeout=timeout) as response:
                if response.status == 200:
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(await response.read())
                    logger.debug(f"成功生成音频文件: {output_path}")
                    return True
                else:
                    logger.error(f"TTS API 请求失败，状态码: {response.status}, 文件: {os.path.basename(output_path)}, 响应: {await response.text()}")
                    return False
        except asyncio.TimeoutError:
            logger.error(f"TTS API 请求超时: {tts_url} for file {os.path.basename(output_path)}")
            return False
        except Exception as e:
            logger.error(f"请求TTS API时发生未知错误: {e}, 文件: {os.path.basename(output_path)}")
            return False

    def _merge_and_cleanup_chunks(self, task_id: str, chapter_number: str, chunk_count: int):
        task_voicedata_dir = os.path.join(config.VOICEDATA_DIR, task_id)
        logger.info(f"开始合并章节 {chapter_number} 的 {chunk_count} 个音频片段...")
        combined_audio = AudioSegment.empty()
        chunk_files_to_remove = []
        for i in range(chunk_count):
            chunk_file_path = os.path.join(task_voicedata_dir, f"{chapter_number}_{i}.wav")
            chunk_files_to_remove.append(chunk_file_path)
            if os.path.exists(chunk_file_path):
                try:
                    segment = AudioSegment.from_wav(chunk_file_path)
                    combined_audio += segment
                except Exception as e:
                    logger.warning(f"无法加载或合并音频片段 {chunk_file_path}: {e}。跳过此片段。")
            else:
                logger.warning(f"音频片段文件不存在: {chunk_file_path}。跳过此片段。")
        final_audio_path = os.path.join(task_voicedata_dir, f"{chapter_number}.wav")
        try:
            combined_audio.export(final_audio_path, format="wav")
            logger.info(f"章节 {chapter_number} 音频合并完成，已保存至: {final_audio_path}")
        except Exception as e:
            logger.error(f"导出合并音频文件失败: {final_audio_path}, 错误: {e}")
            return
        logger.info(f"清理章节 {chapter_number} 的临时音频片段...")
        for file_path in chunk_files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.error(f"删除文件失败: {file_path}, 错误: {e}")

    # --- LangGraph Nodes ---
    def _node_start_processing(self, state: GraphState) -> GraphState:
        metadata_path = state["metadata_path"]
        logger.info(f"Node [1]: Starting process for metadata file: '{os.path.basename(metadata_path)}'")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            task_id = os.path.basename(metadata_path).replace('-metadata.json', '')
            task_voicedata_dir = os.path.join(config.VOICEDATA_DIR, task_id)
            os.makedirs(task_voicedata_dir, exist_ok=True)
            return {**state, "metadata": metadata, "task_id": task_id, "task_voicedata_dir": task_voicedata_dir, "final_audio_paths": [], "processing_start_time": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Failed to read or parse metadata file: {metadata_path}", exc_info=True)
            return {**state, "metadata": {"error": True}}

    async def _node_process_chapters(self, state: GraphState) -> GraphState:
        """节点：遍历章节，使用语义分块，生成并合并音频。"""
        logger.info("Node [2]: Processing all chapters to generate audio...")
        task_id, metadata = state["task_id"], state["metadata"]
        final_audio_paths = []
        semaphore = asyncio.Semaphore(self.tts_config.max_concurrent_requests)

        async with aiohttp.ClientSession() as session:
            for chapter_file_relative_path in metadata.get('split_files', []):
                chapter_file_path = os.path.join(config.SPLITDATA_DIR, task_id, os.path.basename(chapter_file_relative_path))
                chapter_number = os.path.splitext(os.path.basename(chapter_file_path))[0]
                if not os.path.exists(chapter_file_path):
                    logger.warning(f"Chapter file not found: {chapter_file_path}, skipping.")
                    continue
                try:
                    with open(chapter_file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except Exception as e:
                    logger.error(f"Failed to read chapter file: {chapter_file_path}", exc_info=True)
                    continue

                chunks = await self._intelligent_split(text)
                logger.info(f"章节 {chapter_number} 被语义分割成 {len(chunks)} 个块。")

                tts_tasks = []
                for i, chunk in enumerate(chunks):
                    clean_chunk = clean_text_for_tts(chunk)
                    if not clean_chunk: continue
                    output_path = os.path.join(state["task_voicedata_dir"], f"{chapter_number}_{i}.wav")
                    async def do_fetch(sem, text, path):
                        async with sem:
                            return await self._fetch_tts_audio(session, text, path)
                    tts_tasks.append(do_fetch(semaphore, clean_chunk, output_path))

                logger.info(f"正在为章节 {chapter_number} 的 {len(tts_tasks)} 个块生成音频...")
                results = []
                for f in tqdm(asyncio.as_completed(tts_tasks), total=len(tts_tasks), desc=f"Chapter {chapter_number} Audio"):
                    results.append(await f)

                if not all(results):
                    logger.warning(f"章节 {chapter_number} 的部分音频块生成失败，合并结果可能不完整。")

                self._merge_and_cleanup_chunks(task_id, chapter_number, len(chunks))
                final_audio_path = os.path.join('voicedata', task_id, f"{chapter_number}.wav")
                final_audio_paths.append(final_audio_path.replace('\\', '/'))

        return {**state, "final_audio_paths": final_audio_paths}

    def _node_update_metadata(self, state: GraphState) -> GraphState:
        logger.info("Node [3]: Finalizing process and updating metadata file.")
        metadata = state["metadata"]
        metadata['voice_task_id'] = str(uuid.uuid4())
        metadata['audio_files'] = state["final_audio_paths"]
        metadata['audio_processing_start_time'] = state["processing_start_time"]
        metadata['audio_processing_end_time'] = datetime.now().isoformat()
        try:
            with open(state["metadata_path"], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"Task {state['task_id']} metadata updated successfully!")
        except Exception as e:
            logger.critical(f"Failed to write updated metadata for task {state['task_id']}", exc_info=True)
        return state

    def _edge_should_process(self, state: GraphState) -> str:
        if state["metadata"].get("error"): return "end"
        if 'voice_task_id' in state["metadata"]:
            logger.info(f"Task '{state['metadata'].get('source_filename')}' (ID: {state['task_id']}) has already been processed. Skipping.")
            return "end"
        else:
            return "continue"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("start", self._node_start_processing)
        workflow.add_node("process_chapters", self._node_process_chapters)
        workflow.add_node("update_metadata", self._node_update_metadata)
        workflow.set_entry_point("start")
        workflow.add_conditional_edges("start", self._edge_should_process, {"continue": "process_chapters", "end": END})
        workflow.add_edge("process_chapters", "update_metadata")
        workflow.add_edge("update_metadata", END)
        return workflow.compile()

    async def run(self, metadata_path: str):
        initial_state = {"metadata_path": metadata_path}
        await self.app.ainvoke(initial_state)
