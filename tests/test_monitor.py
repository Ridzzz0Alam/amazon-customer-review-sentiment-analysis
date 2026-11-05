import json

from sentiment_mlops.monitor import main


def _write(tmp_path, n_neg, n_pos, words=20):
    log = tmp_path / "p.jsonl"
    rows = [{"endpoint": "single", "label": "negative", "p_pos": 0.05, "n_words": words}] * n_neg
    rows += [{"endpoint": "single", "label": "positive", "p_pos": 0.95, "n_words": words}] * n_pos
    log.write_text("\n".join(json.dumps(r) for r in rows))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"data": {"negative_share_dev": 0.3, "median_words_dev": 20}}))
    return ["--log", str(log), "--meta", str(meta)]


def test_no_drift(tmp_path):
    assert main(_write(tmp_path, 30, 70)) == 0


def test_label_drift(tmp_path):
    assert main(_write(tmp_path, 80, 20)) == 2


def test_length_drift(tmp_path):
    assert main(_write(tmp_path, 30, 70, words=200)) == 2
