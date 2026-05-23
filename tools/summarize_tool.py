"""Summarization tool using ChatPromptTemplate + LLM for text condensation."""
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from utils.llm import llm

_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业新闻编辑。将用户提供的文本总结为简洁摘要：
1. 提取核心事实（5W1H）
2. 保留关键数据与引用
3. 中文输出，300字内
4. 如果是多篇文章，给出对比与综合要点"""),
    ("human", "请总结以下内容（{max_words}字以内）：\n\n{text}"),
])


@tool
def summarize_text(text: str, max_words: int = 300) -> str:
    """调用AI对文本进行总结摘要，提取关键信息。

    Args:
        text: 需要总结的原文内容
        max_words: 摘要最大字数，默认300
    """
    system, user = _prompt.format_messages(text=text, max_words=max_words)
    return llm.ask(str(system.content), str(user.content))
