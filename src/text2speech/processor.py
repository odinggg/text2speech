# src/text2speech/processor.py
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import requests
# [新增] 导入 LangChain 的文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, SecretStr
from pydub import AudioSegment
from tqdm import tqdm

from . import config
from .text_utils import clean_text_for_tts

logger = logging.getLogger(__name__)


# --- Pydantic Models for Configuration ---
class TTSConfig(BaseModel):
    base_url: str = Field(default=config.TTS_BASE_URL)
    endpoint: str = Field(default=config.TTS_ENDPOINT)
    timeout: int = Field(default=config.TTS_REQUEST_TIMEOUT)


class EmbeddingConfig(BaseModel):
    api_key: SecretStr = Field(default=config.EMBEDDING_API_KEY)
    base_url: Optional[str] = Field(default=config.EMBEDDING_BASE_URL)
    model_name: str = Field(default=config.EMBEDDING_MODEL_NAME)
    split_threshold: float = Field(default=config.SEMANTIC_SPLIT_THRESHOLD)


class TextToSpeechProcessor:
    def __init__(self, tts_config: Optional[TTSConfig] = None, embedding_config: Optional[EmbeddingConfig] = None):
        self.tts_config = tts_config or TTSConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        # [新增] 初始化一个基于大小的文本分割器，用于细分过大的块
        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        logger.info(f"处理器已初始化。块大小约束: {config.CHUNK_SIZE}, 重叠: {config.CHUNK_OVERLAP}")

    def _update_metadata(self, metadata_path: str, new_data: Dict):
        """安全地更新元数据文件。"""
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.critical(f"写入元数据文件失败: {metadata_path}", exc_info=True)

    def _get_embeddings(self, sentences: List[str]) -> Optional[np.ndarray]:
        # ... 此方法逻辑不变 ...
        embedding_url = f"{self.embedding_config.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        payload = {"input": sentences, "model": self.embedding_config.model_name}
        try:
            response = requests.post(embedding_url, headers=headers, json=payload, timeout=self.tts_config.timeout)
            response.raise_for_status()
            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
                return np.array(embeddings)
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"请求 Embedding API 时出错: {e}")
            return None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # ... 此方法逻辑不变 ...
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return 0.0 if norm_vec1 == 0 or norm_vec2 == 0 else dot_product / (norm_vec1 * norm_vec2)

    def _intelligent_split(self, text: str) -> List[str]:
        """[修改] 使用混合策略进行文本分割。"""
        logger.info("步骤 1: 开始进行混合文本分块...")

        # --- 阶段 A: 语义分割 ---
        text_norm = re.sub(r'([。！？\?])([^”’])', r"\1\n\2", text)
        text_norm = re.sub(r'(\.{6})([^”’])', r"\1\n\2", text_norm)
        text_norm = re.sub(r'(\…{2})([^”’])', r"\1\n\2", text_norm)
        text_norm = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text_norm)
        sentences = [s.strip() for s in text_norm.split('\n') if s.strip()]

        if len(sentences) <= 1:
            logger.info("文本只有一个句子或更少，跳过语义分割。")
            semantic_chunks = [text]
        else:
            try:
                embeddings = self._get_embeddings(sentences)
                if embeddings is None: raise ValueError("获取 embeddings 失败")

                similarities = [self._cosine_similarity(embeddings[i], embeddings[i + 1]) for i in
                                range(len(embeddings) - 1)]
                semantic_chunks, start_idx = [], 0
                for i, sim in enumerate(similarities):
                    if sim < self.embedding_config.split_threshold:
                        semantic_chunks.append(" ".join(sentences[start_idx: i + 1]))
                        start_idx = i + 1
                semantic_chunks.append(" ".join(sentences[start_idx:]))
                logger.info(f"语义分割完成，得到 {len(semantic_chunks)} 个初始块。")
            except Exception:
                logger.error("语义分块时出错，将整个文本作为一个块处理。", exc_info=True)
                semantic_chunks = [text]

        # --- 阶段 B: 大小约束 ---
        final_chunks = []
        logger.info(f"开始应用大小约束 (最大 {config.CHUNK_SIZE} 字符)...")
        for chunk in semantic_chunks:
            if len(chunk) > config.CHUNK_SIZE:
                logger.info(f"一个语义块长度 ({len(chunk)}) 超出限制，正在进行细分...")
                sub_chunks = self.size_splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
                logger.info(f"  -> 已细分为 {len(sub_chunks)} 个更小的块。")
            else:
                final_chunks.append(chunk)

        logger.info(f"步骤 1 完成: 混合分割完成，最终得到 {len(final_chunks)} 个块。")
        return final_chunks

    # ... 其他方法 (_fetch_tts_audio, _run_splitting_phase, 等) 保持不变 ...
    def _fetch_tts_audio(self, text: str, output_path: str) -> bool:
        """[修改] 带重试逻辑的 TTS 请求方法。"""
        tts_url = f"{self.tts_config.base_url}{self.tts_config.endpoint}"
        form_data = {'text': (None, text)}
        max_retries = 3

        for attempt in range(max_retries):
            logger.info(f"正在尝试第 {attempt + 1}/{max_retries} 次请求 TTS 服务 (内容: '{text[:30]}...')")
            try:
                response = requests.post(tts_url, files=form_data, timeout=self.tts_config.timeout)
                response.raise_for_status()

                if len(response.content) < 1024:
                    logger.warning(f"TTS 响应内容过小 ({len(response.content)} bytes)，可能无效。正在重试...")
                    time.sleep(5)
                    continue

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                if os.path.getsize(output_path) < 1024:
                    logger.warning(f"保存后的 TTS 文件过小，可能已损坏。正在重试...")
                    os.remove(output_path)
                    time.sleep(5)
                    continue

                logger.info(f"成功生成并保存音频文件: {output_path}")
                return True

            except requests.exceptions.RequestException as e:
                logger.error(f"第 {attempt + 1} 次请求 TTS API 时发生 HTTP 错误: {e}")
                if attempt < max_retries - 1:
                    logger.info("将在5秒后重试...")
                    time.sleep(5)
                else:
                    logger.critical("已达到最大重试次数，TTS 转换失败。")
                    return False

        return False

    def _run_splitting_phase(self, metadata_path: str, metadata: Dict) -> bool:
        task_id = metadata['task_id']
        logger.info(f"--- [阶段 1: 文本分割] 开始或继续任务 {task_id} ---")

        vectortemp_dir = os.path.join(config.VECTORTEMP_DIR, task_id)
        os.makedirs(vectortemp_dir, exist_ok=True)
        metadata['vectortemp_dir'] = vectortemp_dir

        for chap_file_rel_path in metadata.get('split_files', []):
            chapter_number = os.path.splitext(os.path.basename(chap_file_rel_path))[0]

            if metadata.get('chapters', {}).get(chapter_number, {}).get('status') == 'split_completed':
                logger.info(f"章节 {chapter_number} 已分割，跳过。")
                continue

            logger.info(f"正在处理章节: {chapter_number}")
            metadata['progress'] = {'current_phase': 'splitting', 'current_chapter': chapter_number}
            self._update_metadata(metadata_path, metadata)

            chapter_file_path = os.path.join(config.SPLITDATA_DIR, task_id, os.path.basename(chap_file_rel_path))
            try:
                with open(chapter_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception:
                logger.error(f"读取章节文件失败: {chapter_file_path}", exc_info=True)
                continue

            chunks = self._intelligent_split(text)

            chapter_chunks_info = []
            for i, chunk_text in enumerate(chunks):
                chunk_filename = f"{chapter_number}-{i}.txt"
                chunk_filepath = os.path.join(vectortemp_dir, chunk_filename)
                with open(chunk_filepath, 'w', encoding='utf-8') as f: f.write(chunk_text)
                chapter_chunks_info.append({'path': chunk_filepath, 'status': 'pending'})

            if 'chapters' not in metadata: metadata['chapters'] = {}
            metadata['chapters'][chapter_number] = {'status': 'split_completed', 'chunks': chapter_chunks_info}
            self._update_metadata(metadata_path, metadata)
            logger.info(f"章节 {chapter_number} 分割完成，保存了 {len(chunks)} 个块。")

        metadata['progress']['current_phase'] = 'tts'
        self._update_metadata(metadata_path, metadata)
        logger.info(f"--- [阶段 1: 文本分割] 任务 {task_id} 全部完成 ---")
        return True

    def _run_tts_phase(self, metadata_path: str, metadata: Dict) -> bool:
        task_id = metadata['task_id']
        logger.info(f"--- [阶段 2: 语音转换] 开始或继续任务 {task_id} ---")

        task_voicedata_dir = os.path.join(config.VOICEDATA_DIR, task_id)
        os.makedirs(task_voicedata_dir, exist_ok=True)

        all_chapters = metadata.get('chapters', {})
        for chapter_number, chapter_data in all_chapters.items():
            if chapter_data.get('status') == 'tts_completed':
                logger.info(f"章节 {chapter_number} 的 TTS 已完成，跳过。")
                continue

            logger.info(f"正在为章节 {chapter_number} 生成语音...")

            all_chunks_successful = True
            for i, chunk_info in enumerate(tqdm(chapter_data['chunks'], desc=f"章节 {chapter_number} TTS")):
                if chunk_info['status'] == 'completed':
                    continue

                metadata['progress'] = {
                    'current_phase': 'tts',
                    'current_chapter': chapter_number,
                    'current_chunk': i
                }
                self._update_metadata(metadata_path, metadata)

                chunk_text_path = chunk_info['path']
                try:
                    with open(chunk_text_path, 'r', encoding='utf-8') as f:
                        chunk_text = f.read()
                except Exception:
                    logger.error(f"读取分块文件失败: {chunk_text_path}", exc_info=True)
                    all_chunks_successful = False
                    break

                clean_chunk = clean_text_for_tts(chunk_text)
                chunk_audio_filename = f"{chapter_number}-{i}.wav"
                chunk_audio_path = os.path.join(task_voicedata_dir, chunk_audio_filename)

                if not self._fetch_tts_audio(clean_chunk, chunk_audio_path):
                    all_chunks_successful = False
                    logger.critical(f"块 {chunk_text_path} TTS 转换在多次重试后最终失败。中止任务 {task_id}。")
                    break
                else:
                    metadata['chapters'][chapter_number]['chunks'][i]['status'] = 'completed'
                    self._update_metadata(metadata_path, metadata)

            if not all_chunks_successful:
                return False

            logger.info(f"正在合并章节 {chapter_number} 的音频...")
            combined_audio = AudioSegment.empty()
            for i in range(len(chapter_data['chunks'])):
                chunk_audio_path = os.path.join(task_voicedata_dir, f"{chapter_number}-{i}.wav")
                if os.path.exists(chunk_audio_path):
                    combined_audio += AudioSegment.from_wav(chunk_audio_path)

            final_audio_path = os.path.join(task_voicedata_dir, f"{chapter_number}.wav")
            combined_audio.export(final_audio_path, format="wav")

            if 'final_audio_files' not in metadata: metadata['final_audio_files'] = []
            relative_final_path = os.path.join('voicedata', task_id, f"{chapter_number}.wav").replace('\\', '/')
            if relative_final_path not in metadata['final_audio_files']:
                metadata['final_audio_files'].append(relative_final_path)

            metadata['chapters'][chapter_number]['status'] = 'tts_completed'
            self._update_metadata(metadata_path, metadata)
            logger.info(f"章节 {chapter_number} 音频合并完成。")

            for i in range(len(chapter_data['chunks'])):
                os.remove(os.path.join(task_voicedata_dir, f"{chapter_number}-{i}.wav"))
            logger.info(f"已清理章节 {chapter_number} 的临时音频文件。")

        metadata['progress'] = {'current_phase': 'completed'}
        metadata['voice_task_id'] = metadata.get('voice_task_id', str(uuid.uuid4()))
        metadata['audio_processing_end_time'] = datetime.now().isoformat()
        self._update_metadata(metadata_path, metadata)
        logger.info(f"--- [阶段 2: 语音转换] 任务 {task_id} 全部完成 ---")
        return True

    def run(self, metadata_path: str):
        logger.info(f"===== 开始处理元数据文件: {os.path.basename(metadata_path)} =====")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.error(f"无法加载或解析元数据: {metadata_path}")
            return

        if 'progress' not in metadata:
            metadata['progress'] = {'current_phase': 'pending'}
        if 'task_id' not in metadata:
            metadata['task_id'] = os.path.basename(metadata_path).replace('-metadata.json', '')
        if 'audio_processing_start_time' not in metadata:
            metadata['audio_processing_start_time'] = datetime.now().isoformat()

        current_phase = metadata['progress']['current_phase']

        if current_phase in ['pending', 'splitting']:
            if not self._run_splitting_phase(metadata_path, metadata):
                logger.error(f"任务 {metadata['task_id']} 在分割阶段失败。")
                return
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            current_phase = metadata['progress']['current_phase']

        if current_phase == 'tts':
            if not self._run_tts_phase(metadata_path, metadata):
                logger.error(f"任务 {metadata['task_id']} 在 TTS 阶段失败，已中止。")
                return

        logger.info(f"===== 任务 {metadata['task_id']} 已成功完成。 =====")
