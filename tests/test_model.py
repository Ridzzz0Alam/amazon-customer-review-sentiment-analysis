import numpy as np

from sentiment_mlops.model import build_pipeline, detect_concerns, explain


def _toy_pipeline():
    texts = [
        "great phone love it",
        "excellent works great",
        "not working broken",
        "terrible battery died",
        "love the screen",
        "stopped working return it",
    ] * 5
    labels = [1, 1, 0, 0, 1, 0] * 5
    return build_pipeline(ngram_range=(1, 2), min_df=1, C=1.0).fit(texts, labels)


def test_pipeline_predicts_direction():
    pipe = _toy_pipeline()
    proba = pipe.predict_proba(["great phone", "broken and not working"])[:, 1]
    assert proba[0] > 0.5 > proba[1]


def test_explain_matches_logit():
    pipe = _toy_pipeline()
    text = "great phone but battery died"
    terms = explain(pipe, text, top_k=100)
    logit = pipe.decision_function([text])[0]
    intercept = pipe.named_steps["clf"].intercept_[0]
    assert np.isclose(sum(t["weight"] for t in terms) + intercept, logit, atol=1e-3)


def test_explain_unknown_words_returns_empty():
    assert explain(_toy_pipeline(), "zzzz qqqq") == []


def test_detect_concerns():
    found = detect_concerns("The battery drains fast and the seller refused a refund")
    assert {"battery", "shipping", "returns"} <= set(found)
    assert detect_concerns("lovely") == []
