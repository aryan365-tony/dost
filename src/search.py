from duckduckgo_search import DDGS

def search_web(query, num_results=3):
    """Perform a live web search using DuckDuckGo (via DDGS)."""
    snippets = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=num_results)
        for res in results:
            snippet = res.get("body") or res.get("snippet") or ""
            if snippet:
                snippets.append(snippet)
    return "\n".join(snippets)
