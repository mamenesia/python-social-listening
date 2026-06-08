import sys
import types

from fastapi.testclient import TestClient

sys.modules.setdefault("google", types.ModuleType("google"))
genai_stub = types.ModuleType("google.generativeai")
genai_stub.GenerativeModel = object
sys.modules.setdefault("google.generativeai", genai_stub)
seaborn_stub = types.ModuleType("seaborn")
seaborn_stub.set_theme = lambda **kwargs: None
sys.modules.setdefault("seaborn", seaborn_stub)

agentic_chat_stub = types.ModuleType("agentic_chat")
agentic_chat_stub.run_agentic_chat = lambda **kwargs: ""
sys.modules.setdefault("agentic_chat", agentic_chat_stub)

wordcloud_stub = types.ModuleType("wordcloud")
wordcloud_stub.WordCloud = object
sys.modules.setdefault("wordcloud", wordcloud_stub)

import main


class FakeSentimentPipeline:
    def __call__(self, texts, **kwargs):
        return [
            {"label": "LABEL_0", "score": 0.9}
            if "bagus" in text.lower()
            else {"label": "LABEL_2", "score": 0.8}
            for text in texts
        ]


async def fake_get_sentiment_pipeline():
    return FakeSentimentPipeline()


def test_sentiment_returns_empty_results_for_empty_texts(monkeypatch):
    monkeypatch.setattr(main, "_get_sentiment_pipeline", fake_get_sentiment_pipeline)
    client = TestClient(main.app)

    response = client.post("/api/v1/sentiment", json={"texts": []})

    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_sentiment_maps_labels_and_preserves_blank_text_order(monkeypatch):
    monkeypatch.setattr(main, "_get_sentiment_pipeline", fake_get_sentiment_pipeline)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/sentiment",
        json={"texts": ["Bagus sekali", "   ", "Sangat buruk"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {"label": "Positive", "score": 0.9},
            {"label": "Neutral", "score": 1.0},
            {"label": "Negative", "score": 0.8},
        ]
    }


def test_sentiment_rejects_batches_over_configured_limit(monkeypatch):
    monkeypatch.setattr(main, "_get_sentiment_pipeline", fake_get_sentiment_pipeline)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/sentiment",
        json={"texts": ["x"] * (main._SENTIMENT_MAX_BATCH + 1)},
    )

    assert response.status_code == 422


def test_sentiment_returns_503_when_model_unavailable(monkeypatch):
    async def failing_get_sentiment_pipeline():
        raise RuntimeError("model failed")

    monkeypatch.setattr(main, "_get_sentiment_pipeline", failing_get_sentiment_pipeline)
    client = TestClient(main.app)

    response = client.post("/api/v1/sentiment", json={"texts": ["Bagus"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "Sentiment model unavailable"}
