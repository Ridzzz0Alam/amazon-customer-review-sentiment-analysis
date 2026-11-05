import pandas as pd

from sentiment_mlops.data import add_labels, collapse_duplicates, group_split, normalise, prepare


def _frame(rows):
    return pd.DataFrame(rows, columns=["Product Name", "Brand Name", "Price", "Rating", "Reviews", "Review Votes"])


def test_normalise_merges_trivial_variants():
    assert normalise("Great!!  Phone") == normalise("great phone") == "great phone"
    assert normalise("___I love it") == "i love it"


def test_add_labels_rules():
    df = _frame(
        [
            ["A", "x", 1, 5, "love it", 0],
            ["A", "x", 1, 4, "nice", 0],
            ["A", "x", 1, 3, "meh", 0],
            ["A", "x", 1, 2, "bad", 0],
            ["A", "x", 1, 1, "awful", 0],
            ["A", "x", 1, 5, "   ", 0],
            ["A", "x", 1, 5, None, 0],
            ["A", "x", 1, 5, "👍", 0],
        ]
    )
    out = add_labels(df)
    assert list(out["Rating"]) == [5, 4, 2, 1]
    assert list(out["label"]) == [1, 1, 0, 0]


def test_collapse_duplicates_majority_and_conflicts():
    rows = [["P1", "x", 1, 5, "good", 0]] * 9 + [["P2", "x", 1, 1, "good", 0]]  # 90% agree -> keep positive
    rows += [["P1", "x", 1, 5, "no", 0], ["P2", "x", 1, 1, "no", 0]]  # 50% -> drop
    rows += [["P3", "x", 1, 1, "broken screen", 0]]
    out = collapse_duplicates(add_labels(_frame(rows)), threshold=0.8)
    assert set(out["text_key"]) == {"good", "broken screen"}
    good = out[out["text_key"] == "good"].iloc[0]
    assert good["label"] == 1 and good["Product Name"] == "P1" and good["copies"] == 10


def test_group_split_has_no_leakage(fixture_csv):
    split = prepare(str(fixture_csv), None, 42, 0.8, 0.2)
    assert not set(split.dev["Product Name"]) & set(split.test["Product Name"])
    assert not set(split.dev["text_key"]) & set(split.test["text_key"])
    assert len(split.dev) > len(split.test) > 0


def test_group_split_is_reproducible(fixture_csv):
    a = prepare(str(fixture_csv), None, 42, 0.8, 0.2)
    b = prepare(str(fixture_csv), None, 42, 0.8, 0.2)
    pd.testing.assert_frame_equal(a.test, b.test)
    assert group_split  # imported for API stability
