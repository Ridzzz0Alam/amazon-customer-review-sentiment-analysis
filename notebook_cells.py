CELLS = []


def md(s):
    CELLS.append(("md", s))


def code(s):
    CELLS.append(("code", s))


md("""
# Customer Sentiment from Amazon Phone Reviews

MD_INTRO
""")

code(r'''
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 200)

RANDOM_STATE = 42
SAMPLE_SIZE = 50_000
DATA_PATH = Path("data/Amazon_Unlocked_Mobile.csv")
FIG_DIR = Path("reports/figures")
MODEL_DIR = Path("models")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Colors: blue for positive and orange for negative in every plot.
POS_COLOR, NEG_COLOR, GREY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
})
print("pandas", pd.__version__, "| scikit-learn", sklearn.__version__)
''')

md("""
## 1. Load the data and draw a reproducible sample

MD_LOAD
""")

code(r'''
raw = pd.read_csv(DATA_PATH)
print("Full dataset shape:", raw.shape)

sample = raw.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
print("Sample shape:", sample.shape)
sample.head()
''')

md("""
## 2. Explore and prepare the data

### 2.1 Missing and empty reviews
""")

code(r'''
missing = sample.isna().sum().rename("missing").to_frame()
missing["share_%"] = (missing["missing"] / len(sample) * 100).round(2)
display(missing)

review_text = sample["Reviews"].fillna("")
is_empty = review_text.str.strip().eq("")
print("Missing review text:", sample["Reviews"].isna().sum())
print("Empty or whitespace only review text:", int(is_empty.sum()))
''')

md("""
MD_MISSING
""")

md("""
### 2.2 Ratings and binary sentiment labels
""")

code(r'''
df = sample.loc[~is_empty].copy()
df["Reviews"] = df["Reviews"].str.strip()

rating_counts = df["Rating"].value_counts().sort_index()

df = df[df["Rating"] != 3].copy()
df["label"] = (df["Rating"] >= 4).astype(int)   # 1 = positive and 0 = negative
label_counts = df["label"].map({1: "positive", 0: "negative"}).value_counts()
print(rating_counts.to_string())
print()
print(label_counts.to_string())
print("Negative share: %.1f%%" % (100 * (df["label"] == 0).mean()))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
colors = [NEG_COLOR, NEG_COLOR, GREY, POS_COLOR, POS_COLOR]
axes[0].bar(rating_counts.index.astype(str), rating_counts.values, color=colors, width=0.7)
axes[0].set_title("Star ratings in the sample (3 stars excluded later)")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Reviews")
for x, v in zip(rating_counts.index.astype(str), rating_counts.values):
    axes[0].text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

axes[1].bar(["negative (1-2)", "positive (4-5)"],
            [label_counts["negative"], label_counts["positive"]],
            color=[NEG_COLOR, POS_COLOR], width=0.6)
axes[1].set_title("Binary sentiment labels")
axes[1].set_ylabel("Reviews")
for i, v in enumerate([label_counts["negative"], label_counts["positive"]]):
    axes[1].text(i, v, f"{v:,} ({v / label_counts.sum():.0%})", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_rating_distribution.png", bbox_inches="tight")
plt.show()
''')

md("""
MD_RATING_PLOT
""")

md("""
### 2.3 Exact duplicates and conflicting labels
""")

code(r'''
def normalise(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace.

    This makes "Great!" and "great" or "___I have owned" and "I have owned" count as
    the same text. Two reviews with the same key are treated as duplicates.
    """
    text = re.sub(r"[\W_]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

df["text_key"] = df["Reviews"].map(normalise)
no_words = df["text_key"].eq("")
print("Reviews with no letters or digits (only emoji or punctuation):", int(no_words.sum()))
print(df.loc[no_words, "Reviews"].value_counts().head(8).to_string())
df = df.loc[~no_words].copy()
print()
dup_stats = df.groupby("text_key").agg(
    copies=("label", "size"),
    n_labels=("label", "nunique"),
    pos_share=("label", "mean"),
    n_products=("Product Name", "nunique"),
)
dups = dup_stats[dup_stats["copies"] > 1]
conflicts = dup_stats[dup_stats["n_labels"] > 1]

print("Rows before de-duplication:", len(df))
print("Unique review texts:", len(dup_stats))
print("Texts that appear more than once:", len(dups), "covering", int(dups["copies"].sum()), "rows")
print("Repeated texts that span more than one product name:", int((dups["n_products"] > 1).sum()))
print("Texts with conflicting labels:", len(conflicts), "covering", int(conflicts["copies"].sum()), "rows")
print()
print("Most repeated texts")
display(dups.sort_values("copies", ascending=False).head(10))
print("Conflicting texts")
display(conflicts.sort_values("copies", ascending=False).head(12))
''')

md("""
MD_DUP_FINDINGS
""")

code(r'''
AGREEMENT_THRESHOLD = 0.80

def collapse_duplicates(frame: pd.DataFrame, threshold: float = AGREEMENT_THRESHOLD) -> pd.DataFrame:
    """Keep one row per normalised text.

    The label is the majority label. Texts whose majority share is below the
    threshold are dropped because their label is close to a coin flip. The kept
    row comes from the most common product name for that text.
    """
    rows = []
    for key, grp in frame.groupby("text_key", sort=False):
        share_pos = grp["label"].mean()
        agreement = max(share_pos, 1 - share_pos)
        if agreement < threshold:
            continue
        majority = int(share_pos >= 0.5)
        top_product = grp["Product Name"].value_counts().index[0]
        keep = grp[(grp["label"] == majority) & (grp["Product Name"] == top_product)]
        if keep.empty:
            keep = grp[grp["label"] == majority]
        row = keep.iloc[0].copy()
        row["copies"] = len(grp)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)

clean = collapse_duplicates(df)
dropped_conflicts = conflicts[conflicts["pos_share"].between(1 - AGREEMENT_THRESHOLD, AGREEMENT_THRESHOLD, inclusive="neither")]
print("Conflicting texts dropped (agreement below 80%):", len(dropped_conflicts))
display(dropped_conflicts)
print("Rows after de-duplication:", len(clean))
print("Negative share after de-duplication: %.1f%%" % (100 * (clean["label"] == 0).mean()))
print("Unique product names:", clean["Product Name"].nunique())
''')

md("""
MD_DUP_POLICY
""")

md("""
### 2.4 Review length by sentiment
""")

code(r'''
clean["n_words"] = clean["Reviews"].str.split().str.len()
length_summary = clean.groupby(clean["label"].map({0: "negative", 1: "positive"}))["n_words"].describe(
    percentiles=[0.25, 0.5, 0.75, 0.95]
).round(1)
display(length_summary)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
bins = np.logspace(0, np.log10(clean["n_words"].max() + 1), 40)
for lab, color, name in [(1, POS_COLOR, "positive"), (0, NEG_COLOR, "negative")]:
    axes[0].hist(clean.loc[clean["label"] == lab, "n_words"], bins=bins, color=color,
                 alpha=0.55, label=name, density=True)
axes[0].set_xscale("log")
axes[0].set_xlabel("Words per review (log scale)")
axes[0].set_ylabel("Density")
axes[0].set_title("Review length distribution")
axes[0].legend(frameon=False)

length_bins = pd.cut(clean["n_words"], [0, 3, 10, 30, 100, 10_000],
                     labels=["1-3", "4-10", "11-30", "31-100", "100+"])
neg_rate = clean.groupby(length_bins, observed=True)["label"].apply(lambda s: (s == 0).mean())
axes[1].bar(neg_rate.index.astype(str), neg_rate.values, color=NEG_COLOR, width=0.65)
axes[1].set_xlabel("Words per review")
axes[1].set_ylabel("Share negative")
axes[1].set_title("Negative share by review length")
for x, v in zip(neg_rate.index.astype(str), neg_rate.values):
    axes[1].text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_review_length.png", bbox_inches="tight")
plt.show()
''')

md("""
MD_LENGTH
""")

md("""
### 2.5 Common words and phrases
""")

code(r'''
def top_terms(texts, ngram_range, k=15):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english", min_df=5)
    counts = vec.fit_transform(texts)
    totals = np.asarray(counts.sum(axis=0)).ravel()
    order = totals.argsort()[::-1][:k]
    return pd.Series(totals[order], index=np.array(vec.get_feature_names_out())[order])

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, (lab, color, name) in enumerate([(1, POS_COLOR, "positive"), (0, NEG_COLOR, "negative")]):
    texts = clean.loc[clean["label"] == lab, "Reviews"]
    for row, ngr in enumerate([(1, 1), (2, 2)]):
        terms = top_terms(texts, ngr)
        ax = axes[row, col]
        ax.barh(terms.index[::-1], terms.values[::-1], color=color, height=0.65)
        ax.set_title(f"Top {'words' if ngr == (1, 1) else 'bigrams'} in {name} reviews")
        ax.set_xlabel("Count")
        ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_common_terms.png", bbox_inches="tight")
plt.show()
''')

md("""
MD_TERMS
""")

md("""
### 2.6 Manual inspection of reviews

Star ratings are a proxy for sentiment. They are not perfect ground truth. Below we read a random set of 24 reviews and then a targeted set of reviews where the text and rating may disagree.
""")

code(r'''
inspect = clean.sample(24, random_state=7)[["Rating", "label", "Reviews"]]
for i, r in enumerate(inspect.itertuples(), 1):
    print(f"[{i:02d}] {r.Rating} stars | {r.Reviews[:260]}")
''')

md("""
MD_RANDOM_INSPECT
""")

code(r'''
positive_cues = r"\b(?:love|great|excellent|perfect|amazing|awesome)\b"
negative_cues = r"\b(?:broken|doesn'?t work|not working|stopped working|defective|refund|return(?:ed)?|scam|terrible|worst)\b"

neg_text_high_star = clean[(clean["label"] == 1) & clean["Reviews"].str.contains(negative_cues, case=False, regex=True)
                           & ~clean["Reviews"].str.contains(positive_cues, case=False, regex=True)]
pos_text_low_star = clean[(clean["label"] == 0) & clean["Reviews"].str.contains(positive_cues, case=False, regex=True)
                          & ~clean["Reviews"].str.contains(negative_cues, case=False, regex=True)
                          & (clean["n_words"] <= 25)]
print("High star reviews with only negative cue words:", len(neg_text_high_star))
print("Short low star reviews with only positive cue words:", len(pos_text_low_star))
print()
print("== Rated 4-5 but the text reads negative ==")
for r in neg_text_high_star.sample(8, random_state=1).itertuples():
    print(f"{r.Rating} stars | {r.Reviews[:240]}")
print()
print("== Rated 1-2 but the text reads positive ==")
for r in pos_text_low_star.sample(8, random_state=1).itertuples():
    print(f"{r.Rating} stars | {r.Reviews[:240]}")
''')

md("""
MD_INCONSISTENT
""")

md("""
## 3. A fair evaluation plan

MD_EVAL_PLAN
""")

code(r'''
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
dev_idx, test_idx = next(splitter.split(clean, clean["label"], groups=clean["Product Name"]))
dev = clean.iloc[dev_idx].reset_index(drop=True)
test = clean.iloc[test_idx].reset_index(drop=True)

summary = pd.DataFrame({
    "reviews": [len(dev), len(test)],
    "product_names": [dev["Product Name"].nunique(), test["Product Name"].nunique()],
    "negative_share": [(dev["label"] == 0).mean().round(3), (test["label"] == 0).mean().round(3)],
}, index=["development", "test"])
display(summary)

shared_products = set(dev["Product Name"]) & set(test["Product Name"])
shared_texts = set(dev["text_key"]) & set(test["text_key"])
print("Product names in both sets:", len(shared_products))
print("Identical normalised texts in both sets:", len(shared_texts))
assert not shared_products and not shared_texts

X_dev, y_dev, g_dev = dev["Reviews"], dev["label"], dev["Product Name"]
X_test, y_test = test["Reviews"], test["label"]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
''')

md("""
MD_SPLIT_RESULT
""")

md("""
## 4. Build and compare models

MD_METRICS
""")

code(r'''
SCORING = {
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "neg_precision": make_scorer(precision_score, pos_label=0, zero_division=0),
    "neg_recall": make_scorer(recall_score, pos_label=0, zero_division=0),
}

def make_lr_pipeline(ngram_range=(1, 1), min_df=2, C=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=min_df,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")),
    ])

def evaluate_cv(name, pipeline):
    res = cross_validate(pipeline, X_dev, y_dev, groups=g_dev, cv=cv, scoring=SCORING, n_jobs=1)
    n_features = None
    if "tfidf" in getattr(pipeline, "named_steps", {}):
        n_features = len(pipeline.fit(X_dev, y_dev).named_steps["tfidf"].vocabulary_)
    row = {"model": name}
    for metric in SCORING:
        vals = res[f"test_{metric}"]
        row[metric] = vals.mean()
        row[f"{metric}_sd"] = vals.std()
    row["fit_seconds"] = res["fit_time"].mean()
    row["predict_seconds"] = res["score_time"].mean()
    row["n_features"] = n_features
    return row

cv_rows = [
    evaluate_cv("Most frequent class", DummyClassifier(strategy="most_frequent")),
    evaluate_cv("LogReg unigrams", make_lr_pipeline((1, 1))),
    evaluate_cv("LogReg unigrams + bigrams", make_lr_pipeline((1, 2))),
]
cv_table = pd.DataFrame(cv_rows).set_index("model")
display(cv_table[["macro_f1", "macro_f1_sd", "accuracy", "neg_precision", "neg_recall",
                  "fit_seconds", "predict_seconds", "n_features"]].round(4))
''')

md("""
MD_BASELINE_COMPARE
""")

md("""
## 5. Tune with GridSearchCV on development data only

MD_GRID_INTRO
""")

code(r'''
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2, 5],
    "clf__C": [0.5, 2.0, 8.0],
}
grid = GridSearchCV(
    make_lr_pipeline(),
    param_grid=param_grid,
    scoring=SCORING,
    refit="macro_f1",
    cv=cv,
    n_jobs=2,
    return_train_score=False,
)
t0 = time.perf_counter()
grid.fit(X_dev, y_dev, groups=g_dev)
print(f"Grid search time: {time.perf_counter() - t0:.0f} s for {len(grid.cv_results_['params'])} settings x 5 folds")

grid_table = pd.DataFrame(grid.cv_results_)
grid_table = grid_table.assign(
    ngram_range=grid_table["param_tfidf__ngram_range"].astype(str),
    min_df=grid_table["param_tfidf__min_df"],
    C=grid_table["param_clf__C"],
)[["ngram_range", "min_df", "C", "mean_test_macro_f1", "std_test_macro_f1", "mean_test_accuracy",
   "mean_test_neg_precision", "mean_test_neg_recall", "mean_fit_time", "rank_test_macro_f1"]]
grid_table = grid_table.sort_values("rank_test_macro_f1").reset_index(drop=True)
display(grid_table.round(4))
print("Best by mean macro F1:", grid.best_params_)
''')

code(r'''
fig, ax = plt.subplots(figsize=(8, 3.8))
markers = {2: "o", 5: "s"}
for ngr, color in [("(1, 1)", GREY), ("(1, 2)", POS_COLOR)]:
    for mdf in [2, 5]:
        sub = grid_table[(grid_table["ngram_range"] == ngr) & (grid_table["min_df"] == mdf)].sort_values("C")
        ax.errorbar(sub["C"], sub["mean_test_macro_f1"], yerr=sub["std_test_macro_f1"], color=color,
                    marker=markers[mdf], markersize=8, linewidth=2, capsize=3,
                    label=f"ngram {ngr}, min_df {mdf}")
ax.set_xscale("log", base=2)
ax.set_xticks([0.5, 2, 8], ["0.5", "2", "8"])
ax.set_xlabel("C (inverse regularisation strength)")
ax.set_ylabel("Mean CV macro F1")
ax.set_title("Grid search on development data (error bars show 1 SD across folds)")
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_grid_search.png", bbox_inches="tight")
plt.show()
''')

md("""
MD_CHOICE
""")

code(r'''
# Selection rule fixed before looking at the test set:
# 1. keep every setting whose mean macro F1 is within one fold SD of the best setting
# 2. among those prefer the smallest vocabulary (cheaper to store and serve)
# 3. then prefer the smaller C (stronger regularisation)
vocab_size = {}
for ngr in param_grid["tfidf__ngram_range"]:
    for mdf in param_grid["tfidf__min_df"]:
        vec = TfidfVectorizer(ngram_range=ngr, min_df=mdf, sublinear_tf=True, strip_accents="unicode")
        vocab_size[(str(ngr), mdf)] = len(vec.fit(X_dev).vocabulary_)
grid_table["n_features"] = [vocab_size[(n, m)] for n, m in zip(grid_table["ngram_range"], grid_table["min_df"])]

best = grid_table.iloc[0]
threshold = best["mean_test_macro_f1"] - best["std_test_macro_f1"]
shortlist = grid_table[grid_table["mean_test_macro_f1"] >= threshold].sort_values(["n_features", "C"])
display(shortlist[["ngram_range", "min_df", "C", "mean_test_macro_f1", "mean_test_neg_recall",
                   "mean_test_neg_precision", "mean_fit_time", "n_features"]].round(4))
chosen = shortlist.iloc[0]
FINAL_PARAMS = {"ngram_range": tuple(int(v) for v in re.findall(r"\d+", chosen["ngram_range"])), "min_df": int(chosen["min_df"]), "C": float(chosen["C"])}

final_pipeline = make_lr_pipeline(**FINAL_PARAMS)
t0 = time.perf_counter()
final_pipeline.fit(X_dev, y_dev)
final_fit_seconds = time.perf_counter() - t0
print("Frozen pipeline:", FINAL_PARAMS)
print(f"Refit on all development data in {final_fit_seconds:.1f} s with "
      f"{len(final_pipeline.named_steps['tfidf'].vocabulary_):,} features")

joblib.dump(final_pipeline, MODEL_DIR / "sentiment_pipeline.joblib")
print("Saved to", MODEL_DIR / "sentiment_pipeline.joblib",
      f"({(MODEL_DIR / 'sentiment_pipeline.joblib').stat().st_size / 1e6:.1f} MB)")
''')

md("""
## 6. Final test evaluation of the frozen pipeline

The pipeline above was saved before this section runs. Nothing below changes the model or its settings.
""")

code(r'''
baseline = DummyClassifier(strategy="most_frequent").fit(X_dev, y_dev)

def test_row(name, model):
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    secs = time.perf_counter() - t0
    return {
        "model": name,
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "accuracy": accuracy_score(y_test, pred),
        "neg_precision": precision_score(y_test, pred, pos_label=0, zero_division=0),
        "neg_recall": recall_score(y_test, pred, pos_label=0, zero_division=0),
        "predict_seconds": secs,
    }

test_table = pd.DataFrame([
    test_row("Most frequent class", baseline),
    test_row("Final LogReg pipeline", final_pipeline),
]).set_index("model")

comparison = cv_table[["macro_f1", "accuracy", "neg_precision", "neg_recall", "fit_seconds"]].copy()
comparison.columns = ["dev_cv_" + c for c in comparison.columns]
best_row = grid_table.iloc[0]
comparison.loc["Tuned LogReg (grid best)"] = [best_row["mean_test_macro_f1"], best_row["mean_test_accuracy"],
                                              best_row["mean_test_neg_precision"], best_row["mean_test_neg_recall"],
                                              best_row["mean_fit_time"]]
print("Development cross-validation (5 group-aware folds)")
display(comparison.round(4))
print("Held-out test set (unseen product names)")
display(test_table.round(4))
''')

code(r'''
test_pred = final_pipeline.predict(X_test)
test_proba = final_pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, test_pred, target_names=["negative (1-2 stars)", "positive (4-5 stars)"], digits=3))

fig, ax = plt.subplots(figsize=(4.8, 4.2))
cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format=",")
ax.set_xlabel("Predicted sentiment")
ax.set_ylabel("Rating based label")
ax.set_title("Confusion matrix on the test set")
ax.grid(False)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_confusion_matrix.png", bbox_inches="tight")
plt.show()
''')

md("""
MD_TEST_RESULTS
""")

md("""
### 6.1 The mistakes that matter most
""")

code(r'''
errors = test.assign(pred=test_pred, p_pos=test_proba)
missed_neg = errors[(errors["label"] == 0) & (errors["pred"] == 1)].sort_values("p_pos", ascending=False)
false_alarm = errors[(errors["label"] == 1) & (errors["pred"] == 0)].sort_values("p_pos")
print(f"Missed negatives (false positives for the positive class): {len(missed_neg)}")
for r in missed_neg.head(10).itertuples():
    print(f"  {r.Rating} stars | P(pos)={r.p_pos:.2f} | {r.Reviews[:200]}")
print()
print(f"False alarms (positive reviews predicted negative): {len(false_alarm)}")
for r in false_alarm.head(10).itertuples():
    print(f"  {r.Rating} stars | P(pos)={r.p_pos:.2f} | {r.Reviews[:200]}")
''')

code(r'''
errors["length_bucket"] = pd.cut(errors["n_words"], [0, 3, 10, 30, 100, 10_000],
                                 labels=["1-3", "4-10", "11-30", "31-100", "100+"])
by_len = errors.groupby("length_bucket", observed=True).apply(
    lambda g: pd.Series({
        "reviews": len(g),
        "error_rate": (g["label"] != g["pred"]).mean(),
        "neg_recall": recall_score(g["label"], g["pred"], pos_label=0, zero_division=0),
    }), include_groups=False)
display(by_len.round(3))
''')

md("""
MD_MISTAKES
""")

md("""
## 7. What customers complain about

MD_CONCERNS_INTRO
""")

code(r'''
vocab = np.array(final_pipeline.named_steps["tfidf"].get_feature_names_out())
coefs = final_pipeline.named_steps["clf"].coef_[0]
order = np.argsort(coefs)
top_neg = pd.Series(coefs[order[:20]], index=vocab[order[:20]])
top_pos = pd.Series(coefs[order[-20:]], index=vocab[order[-20:]])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].barh(top_neg.index[::-1], top_neg.values[::-1], color=NEG_COLOR, height=0.65)
axes[0].set_title("Terms pushing towards negative")
axes[0].set_xlabel("Logistic regression weight")
axes[1].barh(top_pos.index, top_pos.values, color=POS_COLOR, height=0.65)
axes[1].set_title("Terms pushing towards positive")
axes[1].set_xlabel("Logistic regression weight")
for ax in axes:
    ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_model_terms.png", bbox_inches="tight")
plt.show()
''')

code(r'''
CONCERNS = {
    "Battery and charging": r"\bbatter(y|ies)\b|\bcharg(e|er|ing)\b|\bdrain",
    "Screen and display": r"\bscreen\b|\bdisplay\b|\btouch ?screen\b|\bcrack",
    "Network SIM and unlock": r"\bsim\b|\bunlock|\bnetwork\b|\bcarrier\b|\bsignal\b|\b4g\b|\blte\b|\bverizon\b|\bat&t\b|\bt-?mobile\b",
    "Used or refurbished condition": r"\bused\b|\brefurb|\bscratch|\bcondition\b|\bnot new\b",
    "Seller shipping and delivery": r"\bseller\b|\bship(ping|ped)?\b|\bdeliver|\barriv|\bpackag|\bbox\b",
    "Returns refunds and support": r"\breturn|\brefund|\bwarranty\b|\bcustomer service\b|\bsupport\b",
    "Freezing software and speed": r"\bfreez|\bslow\b|\blag|\bsoftware\b|\bupdate|\brestart|\breboot|\bapps?\b",
    "Calls audio and speaker": r"\bcalls?\b|\bspeaker\b|\bmicrophone\b|\bsound\b|\bvolume\b|\bhear\b",
    "Camera": r"\bcamera\b|\bphotos?\b|\bpictures?\b",
    "Fake or wrong item": r"\bfake\b|\bcounterfeit\b|\bwrong\b|\bnot as described\b|\bnot original\b",
}
EXPERIENCE_CONCERNS = {"Seller shipping and delivery", "Returns refunds and support", "Fake or wrong item",
                       "Used or refurbished condition"}

def concern_rates(frame):
    out = {}
    for name, pat in CONCERNS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # groups in the patterns are only for readability
            out[name] = frame["Reviews"].str.contains(pat, case=False, regex=True).mean()
    return pd.Series(out)

rates = pd.DataFrame({
    "negative": concern_rates(clean[clean["label"] == 0]),
    "positive": concern_rates(clean[clean["label"] == 1]),
})
rates["lift"] = rates["negative"] / rates["positive"]
rates["type"] = ["customer experience" if c in EXPERIENCE_CONCERNS else "product" for c in rates.index]
rates = rates.sort_values("negative")
display(rates.sort_values("negative", ascending=False).round(3))

fig, ax = plt.subplots(figsize=(9, 4.6))
y = np.arange(len(rates))
ax.barh(y - 0.2, rates["negative"], height=0.38, color=NEG_COLOR, label="negative reviews")
ax.barh(y + 0.2, rates["positive"], height=0.38, color=POS_COLOR, label="positive reviews")
ax.set_yticks(y, [f"{c} *" if c in EXPERIENCE_CONCERNS else c for c in rates.index])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Share of reviews mentioning the topic")
ax.set_title("Concern topics in negative vs positive reviews (* = seller or service experience)")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "07_concerns.png", bbox_inches="tight")
plt.show()
''')

code(r'''
neg = clean[clean["label"] == 0].copy()
neg["Brand Name"] = neg["Brand Name"].fillna("Unknown")
top_brands = neg["Brand Name"].value_counts().head(6).index
brand_table = pd.DataFrame({b: concern_rates(neg[neg["Brand Name"] == b]) for b in top_brands}).T
brand_table.insert(0, "negative_reviews", neg["Brand Name"].value_counts().loc[top_brands])
display(brand_table.round(2))

voted = neg[neg["Review Votes"].fillna(0) >= 5].sort_values("Review Votes", ascending=False)
voted = voted[~voted["Reviews"].map(normalise).str[:80].duplicated()]   # near copies of one long review
print(f"Negative reviews with 5 or more helpful votes: {len(voted)}")
for _, r in voted.head(5).iterrows():
    print(f"  votes={int(r['Review Votes'])} | {r['Brand Name']} | {r['Reviews'][:220]}")
''')

md("""
MD_CONCERNS_FINDINGS
""")

md("""
### 7.1 Product sentiment versus overall customer experience

MD_PRODUCT_VS_EXPERIENCE
""")

md("""
## 8. Demonstration of the saved model
""")

code(r'''
import sys
sys.path.insert(0, "src")
from predict import load_model, predict_sentiment

model = load_model(MODEL_DIR / "sentiment_pipeline.joblib")
examples = [
    "Great phone. Battery lasts all day and the screen is sharp.",
    "Stopped working after two weeks and the seller never answered my emails.",
    "The phone itself is fine but it arrived three weeks late in a damaged box.",
    "",
    "   ",
    None,
]
for text in examples:
    print(repr(text), "->", predict_sentiment(text, model=model))
''')

md("""
## 9. Conclusions and limitations

MD_CONCLUSIONS
""")
