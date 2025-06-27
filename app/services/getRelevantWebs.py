# app/services/get_relevant_webs.py
import os
import logging
import requests
from dotenv import load_dotenv

class WebSearchService:
    def __init__(self):
        load_dotenv()
        self.duckduckgo_api = os.getenv("DUCKDUCKGO_API")
        self.search_url = "https://www.searchapi.io/api/v1/search"
        if not self.duckduckgo_api:
            raise EnvironmentError("DUCKDUCKGO_API key not set in environment")

    def search(self, query, limit=20):
        """search using DuckDuckGo ."""
        params = {
            "engine": "duckduckgo",
            "q": query,
            "api_key": self.duckduckgo_api,
        }
        headers = {
            "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        }

        resp = requests.get(self.search_url, params=params, headers=headers, timeout=10)

        resp.raise_for_status()
        results = resp.json().get("organic_results", [])
        logging.info(f"DuckDuckGo returned {len(results)} results")
        return [r["link"] for r in results][:limit]
