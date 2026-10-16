import json
import os
import unittest
from unittest.mock import patch

from agents.websearch_agent import WebSearchAgent


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class WebSearchProviderTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.get(key)
            for key in (
                "SHUSHU_SEARCH_ENGINE",
                "SHUSHU_SEARCH_API_KEY",
                "SHUSHU_SEARXNG_URL",
            )
        }

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @patch("agents.websearch_agent.urllib.request.urlopen")
    def test_tavily_provider_returns_parsed_results(self, mock_urlopen):
        payload = json.dumps({
            "results": [
                {"title": "天气", "url": "https://example.com/weather", "content": "北京晴天"},
            ]
        }).encode("utf-8")
        mock_urlopen.return_value = FakeResponse(payload)
        os.environ["SHUSHU_SEARCH_ENGINE"] = "tavily"
        os.environ["SHUSHU_SEARCH_API_KEY"] = "test-key"
        agent = WebSearchAgent.__new__(WebSearchAgent)

        results = agent._provider_search("北京天气")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com/weather")
        self.assertEqual(results[0]["snippet"], "北京晴天")

    def test_provider_without_api_key_falls_back(self):
        os.environ["SHUSHU_SEARCH_ENGINE"] = "tavily"
        os.environ.pop("SHUSHU_SEARCH_API_KEY", None)
        agent = WebSearchAgent.__new__(WebSearchAgent)

        results = agent._provider_search("北京天气")

        self.assertEqual(results, [])

    @patch("agents.websearch_agent.urllib.request.urlopen")
    def test_searxng_provider_parses_json(self, mock_urlopen):
        payload = json.dumps({
            "results": [
                {"title": "新闻", "url": "https://example.com/news", "content": "标题"},
            ]
        }).encode("utf-8")
        mock_urlopen.return_value = FakeResponse(payload)
        os.environ["SHUSHU_SEARCH_ENGINE"] = "searxng"
        os.environ["SHUSHU_SEARXNG_URL"] = "https://search.example.com"
        agent = WebSearchAgent.__new__(WebSearchAgent)

        results = agent._provider_search("新闻")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com/news")


    @patch("agents.websearch_agent.urllib.request.urlopen")
    def test_tavily_quota_falls_back_to_serper(self, mock_urlopen):
        payload = json.dumps({
            "organic": [
                {"title": "News", "link": "https://example.com/news", "snippet": "Headline"},
            ]
        }).encode("utf-8")

        def fake_open(req, *args, **kwargs):
            url = str(getattr(req, "full_url", ""))
            if "api.tavily.com" in url:
                raise RuntimeError("quota exceeded")
            return FakeResponse(payload)

        mock_urlopen.side_effect = fake_open
        os.environ["SHUSHU_SEARCH_ENGINE"] = "tavily"
        os.environ["SHUSHU_SEARCH_API_KEY"] = "test-key"
        agent = WebSearchAgent.__new__(WebSearchAgent)

        results = agent._provider_search("latest news")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com/news")



    @patch.object(WebSearchAgent, "_llm_search_keywords", side_effect=AssertionError("keyword call should be skipped"))
    def test_provider_success_skips_llm_keywords(self, _mock_keywords):
        agent = WebSearchAgent.__new__(WebSearchAgent)
        with patch.object(agent, "_provider_search", return_value=[{"url": "https://example.com"}]):
            results = agent._search("latest news")
        self.assertEqual(len(results), 1)

    @patch.object(WebSearchAgent, "_llm_search_keywords", side_effect=AssertionError("keyword call should be skipped"))
    def test_non_weather_direct_fetch_skips_llm_keywords(self, _mock_keywords):
        agent = WebSearchAgent.__new__(WebSearchAgent)
        self.assertEqual(agent._direct_fetch_fallback("latest news"), {})


if __name__ == "__main__":
    unittest.main()
