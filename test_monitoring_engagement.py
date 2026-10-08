"""Request-level engagement availability: real charts and chat graph, fake LLM only."""
import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_sentiment import main


@pytest.fixture
def chat_module(monkeypatch):
    # test_sentiment stubs the normal import to avoid model/dependency startup.
    spec = importlib.util.spec_from_file_location("monitoring_real_chat", Path(__file__).with_name("agentic_chat.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(main, "run_agentic_chat", module.run_agentic_chat)
    return module


@pytest.mark.parametrize("unavailable", [False, True])
def test_requests_preserve_incomplete_engagement(monkeypatch, chat_module, unavailable):
    prompts = []
    a = {"followers": 400, "posts_scraped": 5,
         "engagement_measured_posts": 0 if unavailable else 2,
         "avg_likes": None if unavailable else 12.5,
         "total_engagement": None if unavailable else 30,
         "sentiment": {"positive": 3, "neutral": 1, "negative": 1}}
    b = {"followers": 200, "posts_scraped": 3, "engagement_measured_posts": 0,
         "avg_likes": None, "total_engagement": None,
         "sentiment": {"positive": 1, "neutral": 2}}
    payload = {"brand_a_name": "Alpha", "brand_a_username": "alpha",
               "brand_b_name": "Beta", "brand_b_username": "beta",
               "brand_a": a, "brand_b": b,
               "comparison": {"engagementTotal": None, "brandAEngagementShare": None},
               "coverage": {"coverageNote": "Caller coverage"},
               "top_posts": [{"caption": "Unmeasured content", "engagement": None}]}

    async def analysis_llm(prompt, **kwargs):
        prompts.append(prompt)
        return '{"executive_summary":"Content and sentiment remain available","content_strategy":"Analyze all retained content"}'

    class LLM:
        def invoke(self, prompt):
            prompts.append(prompt)
            if "Classify the message below" in prompt:
                text = "QUESTION_TYPE: specific\nNEEDS_VISUALIZATION: yes"
            elif "Output ONLY raw Python code" in prompt:
                # Exercise real dataframe and matplotlib capture, without a second mock.
                text = "fig, ax = plt.subplots(); ax.bar(df['brand'], df['positive_sentiment']); plt.show()"
            else:
                text = "Content and sentiment remain available"
            return SimpleNamespace(content=text)

    monkeypatch.setattr(main, "DEEPSEEK_API_KEY", "test-placeholder")
    monkeypatch.setattr(main, "_deepseek_text", analysis_llm)
    monkeypatch.setattr(chat_module, "_llm", LLM)
    client = TestClient(main.app)
    response = client.post("/api/v1/monitoring/analysis", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["executive_summary"] == "Content and sentiment remain available"
    assert body["content_strategy"] == "Analyze all retained content"
    for key in ("engagement_chart_b64", "sentiment_chart_b64"):
        assert base64.b64decode(body[key]).startswith(b"\x89PNG")
    assert f"{a['engagement_measured_posts']}/5 posts measured" in prompts[0]
    assert "3 positive / 1 neutral / 1 negative" in prompts[0]
    assert "Unmeasured content" in prompts[0]
    assert "engagement shares unavailable" in prompts[0]
    assert main.ENGAGEMENT_RULES in prompts[0]

    response = client.post("/api/v1/monitoring/chat", json={"message": "Chart the available metrics", "context": payload})
    assert response.status_code == 200
    assert "Content and sentiment remain available" in response.json()["response"]
    assert "data:image/png;base64," in response.json()["response"]
    assert main.ENGAGEMENT_RULES in prompts[-1]
    assert f"{a['engagement_measured_posts']}/5 posts measured" in prompts[-1]
    chart_prompt = next(p for p in prompts if "Output ONLY raw Python code" in p)
    assert "engagement_measured_posts ONLY" in chart_prompt
    assert "retaining other valid panels" in chart_prompt
    df = chat_module._build_df(payload)
    assert df["posts_scraped"].tolist() == [5, 3]
    assert df["engagement_measured_posts"].tolist() == [a["engagement_measured_posts"], 0]
    assert pd.isna(df.loc[1, "total_engagement"])
    assert df.loc[0, "positive_sentiment"] == 3
    if unavailable:
        assert df["avg_likes"].isna().all()
    else:
        assert df.loc[0, "avg_likes"] == 12.5


def test_compatibility_and_explicit_zero(chat_module):
    assert main.AccountSnapshot().avg_likes is None
    assert main.AccountSnapshot(posts_scraped=4, avg_likes=0).engagement_label("avg_likes") == "0"
    assert main.MonitoringAccountSnapshot(posts_scraped=4).engagement_measured_posts == 4
    snapshot = main.MonitoringAccountSnapshot(posts_scraped=4, engagement_measured_posts=0,
                                              avg_likes=10, total_engagement=40)
    assert snapshot.engagement_label("total_engagement") == "Unavailable (not zero)"
    df = chat_module._build_df({"brand_a": snapshot.model_dump()})
    assert pd.isna(df.loc[0, "total_engagement"])
    assert df.loc[0, "engagement_measured_posts"] == 0
    with pytest.raises(ValueError):
        main.MonitoringAccountSnapshot(avg_likes=-1)


def test_unknown_chart_panels_retain_valid_metrics(monkeypatch):
    figures = []
    monkeypatch.setattr(main, "_fig_to_b64", lambda fig: figures.append(fig) or "chart")
    a = main.MonitoringAccountSnapshot(followers=400, posts_scraped=5, engagement_measured_posts=2,
                                       total_engagement=30, avg_likes=None)
    b = main.MonitoringAccountSnapshot(followers=200, posts_scraped=3,
                                       total_engagement=12, avg_likes=3)
    main._render_engagement_chart_modern(a, b, "Alpha", "Beta")
    followers, engagement, likes = figures[0].axes
    assert [bar.get_height() for bar in followers.patches] == [400, 200]
    assert [bar.get_height() for bar in engagement.patches] == [30, 12]
    assert "partial" in engagement.get_title()
    assert "2/5" in engagement.get_xlabel()
    assert not likes.patches
    assert "Unavailable (not zero)" in likes.texts[0].get_text()
    assert "Beta: 3" in likes.texts[0].get_text()


def test_monitoring_posting_times_use_only_measured_cohort():
    posts = [{"timestamp": "2026-10-01T10:00:00Z", "engagement": value}
             for value in (10, 20, None, -1, 0)]
    summary = main.analyze_posting_times(posts, min_posts=1, measured_only=True)
    assert summary["n"] == 3
    assert summary["best_days"][0][1:] == (10, 3)
    assert main.analyze_posting_times(posts, measured_only=True) is None


@pytest.mark.parametrize("endpoint", ["analysis", "full-analysis"])
@pytest.mark.parametrize("measured", [0, 2])
def test_dashboard_analysis_requests(monkeypatch, endpoint, measured):
    prompts = []

    async def llm(prompt, **kwargs):
        prompts.append(prompt)
        return '{"executive_summary":"All content retained","analysis_text":"All content retained","content_summary":"Unmeasured caption retained"}'

    monkeypatch.setattr(main, "DEEPSEEK_API_KEY", "test-placeholder")
    monkeypatch.setattr(main, "_deepseek_text", llm)
    snapshot = {"followers": 400, "posts_scraped": 5,
                "engagement_measured_posts": measured,
                "avg_likes": 0 if measured else None,
                "total_engagement": 0 if measured else None,
                "sentiment": {"positive": 3, "neutral": 1, "negative": 1}}
    response = TestClient(main.app).post(f"/api/v1/{endpoint}", json={
        "kalventis": snapshot, "gsk": snapshot,
        "posts": [{"caption": "Unmeasured caption retained", "likes": None, "comments": 0}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body.get("executive_summary", body.get("analysis_text")) == "All content retained"
    charts = ([body["engagement_chart_b64"], body["sentiment_chart_b64"]]
              if endpoint == "analysis" else [c["image_base64"] for c in body["charts"]])
    assert len(charts) == 2
    assert all(base64.b64decode(c).startswith(b"\x89PNG") for c in charts)
    competitive = next(p for p in prompts if "KALVENTIS (@kenapaharusvaksin)" in p)
    assert f"{measured}/5 posts measured" in competitive
    assert main.ENGAGEMENT_RULES in competitive
    assert ("Unavailable (not zero)" in competitive) == (measured == 0)
    if measured:
        assert "Avg likes" in competitive
        assert "Engagement 0" in competitive or "Total engagement (measured): 0" in competitive
    if endpoint == "full-analysis":
        assert any("Unmeasured caption retained" in p and main.ENGAGEMENT_RULES in p for p in prompts)


@pytest.mark.parametrize("measured", [0, 2])
def test_dashboard_chat_request(monkeypatch, chat_module, measured):
    prompts = []

    class LLM:
        def invoke(self, prompt):
            prompts.append(prompt)
            if "Classify the message below" in prompt:
                answer = "QUESTION_TYPE: specific\nNEEDS_VISUALIZATION: yes"
            elif "Output ONLY raw Python code" in prompt:
                answer = "fig, ax = plt.subplots(); ax.bar(df['brand'], df['positive_sentiment']); plt.show()"
            else:
                answer = "All content retained"
            return SimpleNamespace(content=answer)

    monkeypatch.setattr(main, "DEEPSEEK_API_KEY", "test-placeholder")
    monkeypatch.setattr(chat_module, "_llm", LLM)
    context = {}
    for brand in ("kalventis", "gsk"):
        context.update({f"{brand}_followers": 400, f"{brand}_posts": 5,
                        f"{brand}_engagement_measured_posts": measured,
                        f"{brand}_avg_likes": 0 if measured else None,
                        f"{brand}_total_engagement": 0 if measured else None,
                        f"{brand}_sentiment": {"positive": 3, "neutral": 1, "negative": 1}})
    response = TestClient(main.app).post("/api/v1/chat", json={
        "message": "Chart my available metrics", "context": context,
    })
    assert response.status_code == 200
    assert "All content retained" in response.json()["response"]
    assert "data:image/png;base64," in response.json()["response"]
    assert main.ENGAGEMENT_RULES in prompts[-1]
    assert f"{measured}/5 posts measured" in prompts[-1]
    assert "3 positive / 1 neutral / 1 negative" in prompts[-1]
    assert ("Total engagement (measured): Unavailable (not zero)" in prompts[-1]) == (measured == 0)
    df = chat_module._build_df({"brand_a": main.SocialListeningContext(**context).snapshot("kalventis").model_dump()})
    assert df.loc[0, "engagement_measured_posts"] == measured
    assert pd.isna(df.loc[0, "total_engagement"]) if not measured else df.loc[0, "total_engagement"] == 0
