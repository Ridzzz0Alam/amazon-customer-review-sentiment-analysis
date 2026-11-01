MD = {}

MD["MD_INTRO"] = """
This notebook builds a sentiment classifier for Amazon reviews of unlocked mobile phones. It also turns the reviews into a short list of concerns a seller can act on.

**Goal.** Predict whether a review is positive or negative from its text alone. Then explain what drives negative reviews.

**Dataset.** [Amazon Reviews Unlocked Mobile Phones](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones) from PromptCloud on Kaggle. It has 413,840 reviews. The columns are Product Name and Brand Name and Price and Rating and Reviews and Review Votes.

**Label rules**

| Rating | Label | Used in the classifier |
|---|---|---|
| 4 or 5 stars | positive (1) | yes |
| 1 or 2 stars | negative (0) | yes |
| 3 stars | neutral | no. Excluded from the core task |

**Workflow**

1. Draw a reproducible sample of 50,000 reviews.
2. Explore ratings and missing text and duplicates and length and vocabulary.
3. Read reviews by hand to see where star ratings and text disagree.
4. Split by product name so the test set only holds products the model never saw.
5. Compare a baseline with two feature sets for one classifier.
6. Tune a small grid with group aware cross validation on development data.
7. Freeze the pipeline and evaluate it once on the test set.
8. Summarise customer concerns for a seller.
9. Save the model and demonstrate a prediction function.

Every random step uses `random_state=42` so the notebook gives the same numbers on every run.
"""

MD["MD_LOAD"] = """
The full file has about 414,000 rows. Experiments on all of it would be slow and would add little for a linear text model. We draw a simple random sample of **50,000 reviews** with a fixed seed. The sample keeps the natural rating mix of the full data.
"""

MD["MD_MISSING"] = """
**What we found.** Only 15 of 50,000 sampled reviews have no text. They carry no signal so we drop them. Brand Name is missing for about 16% of rows and Price for about 1%. Neither column is a model input so we keep those rows. Brand is reported as "Unknown" in the seller analysis.
"""

MD["MD_RATING_PLOT"] = """
**Plot 1. Rating and sentiment distribution.** The left chart shows that five star reviews dominate. One star is the second largest group. Ratings are polarised and few people give two or three stars. After dropping 3 star reviews about 26% of the rows are negative.

**How it affects decisions**

* The classes are imbalanced. A model that always says positive would already get about 74% accuracy. Accuracy alone is therefore misleading. We use **macro F1** as the main metric because it weights both classes equally.
* The negative class is the one a seller cares about. We also report **negative class precision and recall**.
* We set `class_weight="balanced"` in the classifier so that negative errors are not cheap during training.
"""

MD["MD_DUP_FINDINGS"] = """
**What we found.** Duplicates are a big issue in this dataset. About 18,600 rows share their text with at least one other row. Most of them are very short reviews such as "good" or "excelente" or "great phone". Many of them also appear under several product names. This happens because Amazon shows the same review on every colour or storage variant of a listing.

If we split without handling this the same text would sit in both development and test data. The test score would then reward memorising text rather than generalising.

Only 27 repeated texts have conflicting labels. Most conflicts are tiny. For example "good" is positive in 99% of its 735 copies. The rare negative copy is most likely a mismatch between text and stars.
"""

MD["MD_DUP_POLICY"] = """
**Duplicate and conflict policy**

1. Texts are compared after lowercasing and removing punctuation and extra spaces. 55 reviews made only of emoji or punctuation have no words left and are dropped. A bag of words model cannot read them anyway.
2. Each repeated text is collapsed to **one row** before any split. Its label is the majority label.
3. If the majority covers less than 80% of the copies the text is **dropped**. Ten texts such as "no" and "i love" and "pretty good" fall in this group. Their label is close to a coin flip and would only add noise.
4. The kept row is assigned to the product name where the text appears most often. This product name is later used for the group split.

After this step we have 32,728 unique reviews from 3,033 product names. The negative share rises from 26% to 30% because most removed copies were short positive reviews.
"""

MD["MD_LENGTH"] = """
**Plot 2. Review length by sentiment.** Negative reviews are longer. The median negative review has 30 words and the median positive review has 20. The right chart shows that only 17% of reviews with one to three words are negative. That rises to 39% for reviews with 31 to 100 words. Unhappy customers explain what went wrong. Happy customers often write two words.

**How it affects decisions**

* Length is a real signal. TF-IDF with `sublinear_tf=True` dampens repeated words so long reviews do not dominate through raw counts.
* Very short reviews give the model little to work with. We check errors by length after testing.
* We keep long reviews. They are the most informative ones for the seller analysis.
"""

MD["MD_TERMS"] = """
**Plot 3. Common words and phrases.** Positive reviews are dominated by "great" and "love" and "works great" and "battery life". Negative reviews talk about "sim card" and "stopped working" and "waste money" and "touch screen".

The plot also shows an important pitfall. With a standard English stop word list the phrase "doesn't work" becomes "doesn work" and "does work". Words such as "not" and "no" are removed even though they flip the meaning.

**How it affects decisions**

* The model does **not** remove stop words. Negation words are kept so bigrams such as "not working" can be learned.
* Bigrams look useful because many complaints are two word phrases. We test this directly in the model comparison.
* "sim card" and "battery life" appear in both classes. Topic words alone do not reveal sentiment. Context does.
"""

MD["MD_RANDOM_INSPECT"] = """
**Reading 24 random reviews.** Most ratings match the text well. Some cases are worth noting.

* **Mixed reviews.** Review 06 gives 5 stars but complains that the phone keeps saying it does not recognise the SIM card. Review 22 gives 4 stars but is "a bit disappointed". The star rating reflects an overall judgement and the text holds both praise and complaints.
* **Negative ratings that are not about phone quality.** Review 17 says wine damaged the handset and review 08 says the phone broke after a drop. The customer is unhappy so the negative label is correct for sentiment. But a seller should not read these as product defects.
* **Seller and delivery experience.** Review 16 (1 star) complains about a foreign version in an opened box. Review 23 (2 stars) mentions an 18 day delivery and missing features. These are real negative experiences even though they are partly about the seller.
* **Non English text.** Reviews 04 and 07 are Spanish ("Espectacular" and "Genial"). The model can only learn these if they appear often enough in training.

Overall about 21 of the 24 random reviews have a rating that clearly fits the text. The remaining three are mixed rather than wrong.
"""

MD["MD_INCONSISTENT"] = """
**Targeted search for inconsistent reviews.** We searched for high star reviews with only negative cue words and short low star reviews with only positive cue words. This gave 248 and 92 candidates. Reading them shows three kinds of cases.

1. **Clear rating errors.** "Charger for phone not working properly" and "The charger is not working" have 4 stars. A customer probably clicked the wrong star or rated the phone while complaining about the charger. These are true label noise.
2. **Keyword false alarms.** "This was great for a few months then nothing but trouble" has 1 star. It contains "great" but the review is clearly negative. "It is not a great phone" is the same. Here the rating is right and a simple keyword rule is wrong. This shows why we need a model that sees word context.
3. **Mixed reviews.** "Everything except the battery life is great" (2 stars) and the 5 star review listing pros and cons. The rating summarises a trade off. Neither label is wrong.

**Conclusion.** Star ratings are a good but noisy proxy. A small share of labels are wrong. That sets a ceiling on measured accuracy. Some "errors" on the test set will be cases where the model reads the text correctly and the star rating disagrees.
"""

MD["MD_EVAL_PLAN"] = """
**What this evaluation measures.** The test set answers one question. *How well does the model predict sentiment for reviews of phones whose product name never appeared in development data?* That is the realistic use case. A seller lists a new phone and wants to monitor its reviews from day one.

**Why a random split is not enough.** Reviews of the same product share vocabulary such as model names and specific defects. With a random split the model could score well by recognising products instead of sentiment. Splitting by product name prevents that.

**The plan**

1. Collapse duplicate texts first (done above) so identical text cannot cross the split.
2. Hold out about 20% of **product names** with `GroupShuffleSplit`. All reviews of a held out product go to the test set.
3. Use the remaining development data for every decision. That includes preprocessing choices and model comparison and tuning.
4. Inside development data use `StratifiedGroupKFold` with 5 folds and product name as the group. Each validation fold again contains only unseen products. Stratification keeps the class balance similar across folds.
5. Touch the test set once at the end with a frozen pipeline.

**Limitation.** Product names are not perfect groups. The same phone in two colours has two product names. Some similar products can therefore sit on both sides. The test score is still far more honest than a random split.
"""

MD["MD_SPLIT_RESULT"] = """
The split gives 26,668 development reviews from 2,426 products and 6,060 test reviews from 607 products. Both sets have about 30% negative reviews. The checks confirm that no product name and no normalised text appears in both sets.
"""

MD["MD_METRICS"] = """
**Metrics**

* **Macro F1** is the main metric. It averages the F1 score of both classes so the minority negative class counts as much as the positive class.
* **Accuracy** is reported for context. It is easy to read but flattered by imbalance.
* **Negative precision** answers "when the model flags a complaint is it really one".
* **Negative recall** answers "what share of real complaints does the model catch". For a seller this is the most important number because a missed complaint is a missed problem.
* **Training cost** is measured as mean fit time per fold and vocabulary size.

**Models**

* A **most frequent class baseline** that always predicts positive.
* **Logistic regression** on TF-IDF features with **unigrams** only.
* The same logistic regression with **unigrams and bigrams**.

Logistic regression is a strong and fast linear model for sparse text. Its weights are also easy to explain to a seller. Every preprocessing step lives inside a scikit-learn `Pipeline`. The TF-IDF vocabulary is therefore refit on the training part of each fold and never sees the validation fold.
"""

MD["MD_BASELINE_COMPARE"] = """
**Results**

* The baseline gets 70% accuracy but a macro F1 of only 0.41. It never finds a single negative review. This confirms that accuracy alone would be misleading.
* Unigram logistic regression jumps to a macro F1 of 0.903 with negative recall of 0.918.
* Adding bigrams improves macro F1 to 0.911 and negative precision from 0.822 to 0.838. The gain is larger than one fold standard deviation. Both models were scored on the same folds so the comparison is paired and fair. Bigrams capture phrases such as "not working" and "waste money" that single words miss.
* The cost is a vocabulary about nine times larger (about 101,000 features versus 11,500). Training takes about 1.5 seconds per fold instead of 0.6. Both are cheap in absolute terms.
"""

MD["MD_GRID_INTRO"] = """
We run a small grid with 12 settings. Each one is scored with the same 5 group aware folds.

| Parameter | Values | Why it matters |
|---|---|---|
| `ngram_range` | unigrams only and unigrams plus bigrams | Confirms the bigram result inside the tuned setting |
| `min_df` | 2 and 5 | Drops rare terms. A higher value gives a smaller and more robust vocabulary |
| `C` | 0.5 and 2 and 8 | Inverse regularisation strength. Higher C fits the training data more closely |

The grid is refit on macro F1. Negative precision and recall are tracked as extra scores.
"""

MD["MD_CHOICE"] = """
**Plot 4. Grid search results.** Bigram settings (blue) beat unigram settings (grey) at every C. Moving from C = 0.5 to C = 2 gives a clear gain. Moving from 2 to 8 adds very little and the error bars overlap. The choice of `min_df` makes almost no difference.

**How we choose the final pipeline.** We fix a rule before seeing any test result.

1. Keep every setting within one fold standard deviation of the best mean macro F1.
2. Among those prefer the smallest vocabulary.
3. Then prefer the smaller C which means stronger regularisation.

Four bigram settings pass the first step. The rule picks **unigrams + bigrams with `min_df=5` and `C=2`**.

* **Performance.** Its macro F1 is 0.916 against 0.919 for the top setting. The gap is less than half a standard deviation. Its negative recall is 0.920 against 0.907 so it catches more complaints. That matters most for a seller.
* **Simplicity.** Its vocabulary has about 37,000 terms instead of 101,000. Stronger regularisation also makes it less likely to overfit rare phrases.
* **Practical constraints.** The saved model is about 1 MB. It trains in about 2 seconds and scores thousands of reviews per second on a laptop CPU. That is easy to deploy and easy to retrain.

The chosen pipeline is refit on all development data and saved to disk before we look at the test set.
"""

MD["MD_TEST_RESULTS"] = """
**Test results for the frozen pipeline**

| Metric | Development CV (chosen setting) | Test set |
|---|---|---|
| Macro F1 | 0.916 | **0.918** |
| Accuracy | 0.928 | 0.929 |
| Negative precision | 0.851 | 0.852 |
| Negative recall | 0.920 | 0.926 |

The test score closely matches cross validation. This suggests that group aware validation gave an honest estimate and that we did not overfit the development data.

**Plot 5. Confusion matrix.** Of 1,833 negative test reviews the model catches 1,698 and misses 135. Of 4,227 positive reviews it wrongly flags 295 as negative. Because of the balanced class weight the model leans towards flagging complaints. It raises about two false alarms for every missed complaint. For monitoring that is the right trade off. A false alarm costs a seller a few seconds of reading. A missed complaint can hide a defect.
"""

MD["MD_MISTAKES"] = """
**Which mistakes matter most**

1. **Missed negatives (135 reviews).** These are the costly errors. Several groups stand out.
   * *Mild or polite complaints.* "ITs ok for the price" and "smaller than expected" and "Acciptable good" got 1 or 2 stars. The words are neutral or positive so the model reads them as positive.
   * *Praise followed by a deal breaker.* "Product was in great condition however the battery life is non existent" and "One of the most user unfriendly phones ... The price was good. Charged super fast." The positive words outweigh the single complaint in a bag of words model.
   * *Text and rating mismatch.* "The best i phone i by it thanx man" and "I am happy with what I buy" have 1 star. The text is positive so the model is arguably right and the label is noisy.
2. **False alarms (295 reviews).** Many are 4 or 5 star reviews whose text is plainly negative. Examples are "Didnt work from the day we got it!! Not impressed!" and "Did not include note pen! Very disappointed." Here the model reads the text better than the star rating does. Others describe a problem that was later solved by the seller.
3. **Very short reviews.** Negative recall drops to 0.77 for reviews of one to three words. There is little context and words such as "ok" or "no" are ambiguous.

**What this means.** A large share of the remaining errors come from label noise or genuinely mixed reviews rather than from obvious model failures. Better gains would need aspect level labels or a model that reads word order such as a fine tuned transformer.
"""

MD["MD_CONCERNS_INTRO"] = """
A seller does not only want a sentiment score. They want to know *why* customers are unhappy. We combine two views.

1. **Model weights.** The logistic regression weights show which words and phrases push a review towards negative or positive.
2. **Concern topics.** A transparent keyword dictionary groups negative reviews into common topics. We compare how often each topic appears in negative versus positive reviews. Each topic is labelled as a *product* issue or a *customer experience* issue (seller or shipping or returns).
"""

MD["MD_CONCERNS_FINDINGS"] = """
**Plot 6. Model weights.** The strongest negative signal by far is "not". Other negative terms describe failure and regret such as "disappointed" and "slow" and "return" and "defective" and "locked" and "waste". Positive terms are general praise such as "great" and "love" and "excellent" and "perfect".

Two positive terms prove the value of bigrams. "no problems" and "not bad" carry strong positive weights even though "no" and "not" alone are negative. A unigram model cannot learn this.

**Plot 7. Concern topics.** The chart compares how often each topic is mentioned in negative and positive reviews.

* **Returns and refunds and support** show up in 22% of negative reviews but only 3.5% of positive ones. That is six times more often. Once a customer needs to return a phone the experience usually turns negative.
* **Network and SIM and unlock problems** are the most frequent topic at 22%. Many "unlocked" phones do not work with the buyer's carrier or only on 3G.
* **Battery and charging** (21%) and **freezing or slow software and restarts** (17%) are the main hardware and software complaints.
* **Fake or wrong items** are rare (3%) but appear three times as often in negative reviews. They also attract the most helpful votes. The top voted negative reviews describe fake iPhones and Galaxy phones.
* **Shipping and delivery** words appear equally often in both classes. Delivery is mentioned a lot but is not by itself a driver of negative sentiment.

**By brand.** BLU has the most software and screen complaints (26% and 21% of its negative reviews). Apple and Samsung have more complaints about refurbished condition and fakes and network locks. These are typical of resold flagship phones.

**Actions for a seller**

1. Check carrier and band compatibility before listing and state it clearly.
2. Test the battery and charger on refurbished stock.
3. Make returns easy and fast because that topic has the strongest link to negative reviews.
4. Verify authenticity of high value phones.
"""

MD["MD_PRODUCT_VS_EXPERIENCE"] = """
**Label noise** means the star rating does not reflect what the customer wrote. "The charger is not working" with 4 stars is label noise. The text and the label disagree.

**A negative customer experience** is different. The customer is truly unhappy but the cause is not the phone itself. Examples are a late delivery or a box that arrived opened or a seller who never replied. "It arrived three weeks late in a damaged box" with 1 star is **not** label noise. The text and rating agree. The customer had a bad experience.

The model predicts **overall customer sentiment**. That is what the star rating measures. A seller or delivery complaint is therefore a correct negative example and we keep it in training. Removing it would teach the model that delivery problems are neutral.

For decisions the seller needs a second layer. The concern topics above separate **product sentiment** (battery and screen and software) from **experience sentiment** (shipping and returns and authenticity). This tells the seller whether to fix the phone or fix the process. For example a 1 star review about a wine damaged handset is negative sentiment but says nothing about product quality. That is why topic analysis sits next to the classifier rather than replacing it.
"""

MD["MD_CONCLUSIONS"] = """
**Summary**

| Model | Dev CV macro F1 | Test macro F1 | Test negative recall |
|---|---|---|---|
| Most frequent class | 0.412 | 0.411 | 0.000 |
| LogReg unigrams | 0.903 | not tested | not tested |
| LogReg unigrams + bigrams | 0.911 | not tested | not tested |
| **Final LogReg (1,2) min_df=5 C=2** | **0.916** | **0.918** | **0.926** |

Only the baseline and the frozen final pipeline were scored on the test set. The other models were compared on development data only.

**Key findings**

* Handling duplicates was essential. About 40% of rows were repeated text copied across product variants.
* A TF-IDF logistic regression with unigrams and bigrams reaches a macro F1 of about 0.92 on unseen products. It catches about 93% of negative reviews.
* Development and test scores agree closely so the evaluation plan was fair.
* Returns and carrier compatibility are the clearest sources of unhappy customers. Battery and software follow.

**Limitations**

* Star ratings are a noisy proxy. Some test "errors" are label errors.
* 3 star reviews are excluded. In practice a seller would also want a neutral or mixed class.
* Product names are imperfect groups because colour variants have different names.
* Near duplicates that differ by a few words are not merged.
* A bag of words model cannot follow "great but" constructions. A transformer model would likely do better on mixed reviews at a higher serving cost.
* Concern topics come from a hand written keyword list. They are transparent but will miss some phrasings.

**Reproducibility.** The sample and split and folds all use `random_state=42`. The final pipeline is saved in `models/sentiment_pipeline.joblib`. It can be used with `src/predict.py` as described in the README.
"""
