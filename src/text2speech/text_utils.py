# text_utils.py
import re

def clean_text_for_tts(text: str) -> str:
    """
    通过正则去除可能影响TTS的特殊符号或非标准字符，
    保留对语音合成有意义的标点符号。

    Args:
        text: 原始文本片段。

    Returns:
        清理后的文本。
    """
    if not isinstance(text, str):
        return ""

    # 1. 替换连续的空白符（包括换行、制表符）为一个空格
    text = re.sub(r'\s+', ' ', text)

    # 2. 移除可能导致问题的特殊字符，但保留基本的中英文标点和字母数字。
    # 这个正则表达式保留了：
    # - \w: 单词字符 (a-z, A-Z, 0-9, _)
    # - \u4e00-\u9fa5: 中文字符
    # - ，。？！、：；“”《》: 中文常用标点
    # - ,.?!:;'" : 英文常用标点
    text = re.sub(r'[^\w\u4e00-\u9fa5\s，。？！、：；“”《》,.?!:;\'"]', '', text)

    # 3. 去除首尾的空白或标点
    text = text.strip()

    return text
