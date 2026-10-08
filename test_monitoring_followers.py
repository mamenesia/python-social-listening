"""Monitoring must preserve unavailable and rounded follower evidence."""
import base64

import pytest
from fastapi.testclient import TestClient

# Reuse the existing lightweight service import (no model downloads).
from test_sentiment import main


@pytest.mark.parametrize("count,approximate,label", [
    (416, False, "416"),
    (0, False, "0"),
    (2700, True, "Approximately 2,700"),
    (None, False, "Unavailable (not zero)"),
])
def test_monitoring_accepts_follower_evidence(monkeypatch, count, approximate, label):
    captured = {}

    async def generate(prompt, **kwargs):
        captured["prompt"] = prompt
        return '{"executive_summary":"Analyzed engagement and sentiment"}'

    def chat(**kwargs):
        captured.update(kwargs)
        return "Analyzed engagement and sentiment"

    monkeypatch.setattr(main, "DEEPSEEK_API_KEY", "test-placeholder")
    monkeypatch.setattr(main, "_deepseek_text", generate)
    monkeypatch.setattr(main, "run_agentic_chat", chat)
    payload = {
        "brand_a_name": "Alpha", "brand_a_username": "alpha",
        "brand_b_name": "Beta", "brand_b_username": "beta",
        "brand_a": {"followers": count, "followers_approximate": approximate,
                    "posts_scraped": 11,
                    "total_engagement": 137, "avg_likes": 19,
                    "sentiment": {"positive": 7, "neutral": 3, "negative": 1}},
        "brand_b": {"followers": 913, "posts_scraped": 1, "total_engagement": 61},
        "comparison": {},
        "coverage": {},
    }
    client = TestClient(main.app)
    result = client.post("/api/v1/monitoring/analysis", json=payload)
    assert result.status_code == 200
    body = result.json()
    assert body["executive_summary"] == "Analyzed engagement and sentiment"
    assert base64.b64decode(body["engagement_chart_b64"]).startswith(b"\x89PNG")
    assert body["sentiment_chart_b64"]
    assert f"Followers: {label}" in captured["prompt"]
    assert main.FOLLOWER_RULES in captured["prompt"]
    result = client.post("/api/v1/monitoring/chat", json={"message": "Compare", "context": payload})
    assert result.status_code == 200
    assert result.json()["response"] == "Analyzed engagement and sentiment"
    assert f"Followers: {label}" in captured["context_block"]
    assert main.FOLLOWER_RULES in captured["context_block"]
    brand = captured["context_data"]["brand_a"]
    assert brand["followers"] == count
    assert brand["followers_approximate"] == approximate
    assert brand["total_engagement"] == 137
    assert brand["sentiment"].positive == 7


def test_missing_followers_default_to_unknown():
    assert main.MonitoringAccountSnapshot().followers is None
    with pytest.raises(ValueError):
        main.MonitoringAccountSnapshot(followers=-1)


@pytest.mark.parametrize("count,approximate", [(None, False), (2700, True)])
def test_chart_preserves_other_metrics(monkeypatch, count, approximate):
    figures = []
    monkeypatch.setattr(main, "_fig_to_b64", lambda fig: figures.append(fig) or "chart")
    a = main.MonitoringAccountSnapshot(followers=count, followers_approximate=approximate,
                                       posts_scraped=1, total_engagement=137, avg_likes=19)
    b = main.MonitoringAccountSnapshot(followers=913, posts_scraped=1, total_engagement=61, avg_likes=5)
    main._render_engagement_chart_modern(a, b, "Alpha", "Beta")
    followers, engagement, likes = figures[0].axes
    assert [bar.get_height() for bar in engagement.patches] == [137, 61]
    assert [bar.get_height() for bar in likes.patches] == [19, 5]
    if count is None:
        assert not followers.patches
        assert "Unavailable (not zero)" in followers.texts[0].get_text()
        assert "913" in followers.texts[0].get_text()
    else:
        assert followers.get_title() == "Followers (approximate)"
        assert [bar.get_height() for bar in followers.patches] == [2700, 913]
