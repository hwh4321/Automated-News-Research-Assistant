"""Report generation tool — creates structured markdown research reports."""
from typing import Any

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from utils.llm import llm

_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名资深研究分析师。根据提供的搜索材料和摘要，生成一份结构化的研究报告：

## 报告结构
1. **标题**：简明扼要的研究标题
2. **摘要**：200字内的核心发现
3. **关键发现**：3-5个要点，每个带有来源引用
4. **详细分析**：分主题深入分析
5. **趋势与展望**：未来发展方向
6. **信息来源**：列出所有参考链接

要求：专业、客观、数据驱动，中文输出，Markdown格式。"""),
    ("human", """研究主题：{topic}

## 搜索材料
{materials_text}

## 已有摘要
{summaries_text}

请根据以上材料生成一份完整的研究报告。"""),
])


def _build_materials_text(materials: list[dict[str, Any]]) -> str:
    return "\n\n---\n".join(
        f"**{m['title']}**\n来源: {m.get('url') or m.get('source', '')}\n{m.get('snippet', '')}"
        for m in materials
    )


@tool
def generate_report(topic: str, materials: Any) -> str:
    """根据收集的新闻材料生成一份结构化的Markdown研究报告。

    Args:
        topic: 研究主题
        materials: 搜索材料列表，每条材料包含title、url/source、snippet字段
    """
    if isinstance(materials, str):
        import json
        try:
            materials = json.loads(materials)
        except json.JSONDecodeError:
            materials = []

    materials_text = _build_materials_text(materials)
    system, user = _prompt.format_messages(
        topic=topic,
        materials_text=materials_text,
        summaries_text="无",
    )
    return llm.ask(str(system.content), str(user.content), temperature=0.4)
