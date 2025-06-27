# getRelevantWebs.py

import os
from dotenv import load_dotenv
import requests

class webContent:
    def __init__(self):
        self.url = "https://www.searchapi.io/api/v1/search"
        self.engine = "duckduckgo"
        load_dotenv()

    def getWebContents(self,description):
        params = {
            "engine":self.engine,
            "q":description,
            "api_key": os.getenv("DUCKDUCKGO_API")
        }

        try:
            response = requests.get(self.url,params=params)

            if response.status_code == 200:
                return response.json()
            else:
                return {"Error": f"Search failed | Status code: {response.status_code}"}
        except Exception as e:
            return {"Error": f"Failed to search from DuckDuckGo | {e}"}

    def toJsonObject(self, response):
        overview = response.get("organic_results", [])
        return [{"title": item["title"], "link": item["link"],"content":item["snippet"]} for item in overview]





getWeb = webContent()
print(getWeb.toJsonObject(getWeb.getWebContents("what is chatgpt")))