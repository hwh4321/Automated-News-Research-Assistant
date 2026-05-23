"""News and web search via SerpAPI / NewsAPI / Tavily."""
from typing import Any
import requests

from langchain_core.tools import tool

from config import config

_ENGINES: dict[str, str] = {
    "tavily": "https://api.tavily.com/search",
    "newsapi": "https://newsapi.org/v2/everything",
    "serpapi": "https://serpapi.com/search",
}


@tool
def search_news(query: str, num: int = 10) -> list[dict[str, Any]]:
    """搜索互联网上的最新新闻和资讯。输入搜索关键词，返回相关新闻文章列表。

    Args:
        query: 搜索关键词或查询语句
        num: 返回结果数量，默认10条
    """
    match config.search_engine:
        case "tavily":
            return _search_tavily(query, num)
        case "newsapi":
            return _search_newsapi(query, num)
        case _:
            return _search_serpapi(query, num)


def _fetch(url: str, **kwargs: Any) -> dict[str, Any]:
    resp = requests.request(kwargs.pop("method", "GET"), url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _search_tavily(query: str, num: int) -> list[dict[str, Any]]:
    data = _fetch(_ENGINES["tavily"], method="POST", json={
        "api_key": config.search_api_key, "query": query,
        "max_results": num, "include_raw_content": False,
    })
    return [
        {"title": r["title"], "url": r["url"],
         "snippet": r.get("content", ""), "source": "tavily"}
        for r in data.get("results", [])[:num]
    ]


def _search_newsapi(query: str, num: int) -> list[dict[str, Any]]:
    data = _fetch(_ENGINES["newsapi"], params={
        "apiKey": config.search_api_key, "q": query,
        "pageSize": num, "language": "zh",
    })
    return [
        {"title": a["title"], "url": a["url"],
         "snippet": a.get("description", ""),
         "source": (a.get("source") or {}).get("name", ""),
         "published": a.get("publishedAt", "")}
        for a in data.get("articles", [])[:num]
    ]


def _search_serpapi(query: str, num: int) -> list[dict[str, Any]]:
    data = _fetch(_ENGINES["serpapi"], params={
        "api_key": config.search_api_key, "q": query, "num": num, "engine": "google",
    })
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""),
         "snippet": r.get("snippet", ""), "source": "serpapi"}
        for r in data.get("organic_results", [])[:num]
    ]
