"""QA Chain — modern LCEL-based RAG: retriever | format_docs | prompt | LLM | StrOutput.

No create_stuff_documents_chain / create_retrieval_chain helpers — just LCEL pipes.
"""
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.llm import llm
from memory.vector_store import vector_store

_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个研究助手，根据提供的参考资料回答用户问题。

规则：
1. 仅根据参考资料作答，不要编造信息
2. 如果参考资料不足以回答问题，明确说明
3. 引用时标注来源（标题、URL）
4. 中文回答，结构化呈现
5. 如果问题涉及多个文档，进行对比和综合"""),
    ("human", "参考资料：\n\n{context}\n\n用户问题：{question}"),
])


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n".join(
        f"[来源 {i+1}] {d.metadata.get('topic', '未知主题')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


class _VectorStoreRetriever(BaseRetriever):
    """LangChain retriever wrapping the project's ChromaDB vector store."""
    k: int = 5

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return [
            Document(page_content=d["text"][:4000], metadata=d["metadata"])
            for d in vector_store.query(query, n_results=self.k)
        ]


_retriever = _VectorStoreRetriever(k=5)

_rag_chain = (
    {"context": _retriever | _format_docs, "question": RunnablePassthrough()}
    | _prompt
    | llm.with_temp(0.2)
    | StrOutputParser()
)


@tool
def ask_research(question: str, top_k: int = 5) -> str:
    """基于历史研究资料回答问题，使用RAG检索增强生成。先检索相关文档，再基于文档生成答案。

    Args:
        question: 用户的问题
        top_k: 检索相关文档数量，默认5
    """
    _retriever.k = top_k
    return _rag_chain.invoke(question)
