# Customer Segmentation + Next-Best-Offer Engine

An end-to-end customer analytics project on a real two-year retail transaction dataset. It groups customers into behavioural segments, recommends the next products to offer each customer, and tests those recommendations honestly against what customers actually bought afterwards.

**Headline result:** an item-item collaborative filtering recommender hit about **53% more** of customers' real future purchases than a "recommend the best sellers" baseline (precision@3 of 6.83% vs 4.47% among customers who bought in the holdout window). A segment-based propensity approach built first did **not** beat the baseline, and this README explains why that matters.

Author: Kamal Ahmad · github.com/kamaal099

---

## 1. The business problem

An online retailer sends the same offers to everyone. That wastes marketing spend on customers who were never going to respond, and it misses customers who are drifting away. Two questions follow:

1. **Who are our customers?** Which groups behave differently enough to be treated differently?
2. **What should we offer each one next?** Which products is a given customer most likely to buy in the near future?

The project answers both with data, and measures whether the answers are actually better than a simple default.

## 2. Data

**UCI Online Retail II**, a real transaction log from a UK-based online retailer that mostly sells gift-ware, with many wholesale customers.

| Item | Value |
|---|---|
| Rows | 1,067,371 transactions |
| Period | 1 Dec 2009 to 9 Dec 2011 |
| Fields | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |
| Licence | CC BY 4.0 |
| Source | https://archive.ics.uci.edu/dataset/502/online+retail+ii |

The dataset is not included in this repository. Download `online_retail_II.xlsx` from the link above and place it in `data/raw/`. No synthetic or generated data is used anywhere in this project.

## 3. What was built, step by step

1. **Ingestion and cleaning (`data.py`).** Reads CSV, XLS or XLSX (all sheets), normalises column names, applies cleaning rules, and keeps a cleaning log. Cancellations and returns are retained in audit metadata rather than silently discarded. A cleaned copy is cached as parquet so later runs are fast.
2. **Leakage-safe cutoff (`features.py`).** The cutoff is the last transaction date minus 90 days. Everything before it is "history" used to build features and recommendations. Everything after it is the "holdout" used only for scoring.
3. **Customer features.** Recency, frequency and monetary value (RFM), plus total quantity, number of distinct products and average order value, built from history only.
4. **Segmentation (`segment.py`).** Clusters customers into three segments and produces a profile table for each.
5. **Propensity model (`propensity.py`) and segment engine (`recommend.py`).** A first recommender that ranks products by segment-level purchase propensity, excluding items a customer already bought.
6. **Item-item recommender (`itemitem.py`).** Builds a customer-by-product purchase matrix from history, computes cosine similarity between products, and recommends the three unseen products most similar to what each customer already bought.
7. **Evaluation (`evaluate.py`, `scripts/baseline.py`).** Scores every method against real purchases in the 90-day holdout using precision@3, recall@3 and NDCG@3, and compares all of them with a popularity baseline.

## 4. Results

### 4.1 Customer segments

| Segment | Customers | Share | Recency (days since last order) | Orders | Spend (per customer) | Distinct products | Avg order value |
|---|---|---|---|---|---|---|---|
| Core customers | 2,882 | 54.7% | 75 | 7.8 | 3,199 | 103 | 403 |
| At risk | 2,358 | 44.8% | 372 | 2.1 | 643 | 34 | 307 |
| High value | 27 | 0.5% | 75 | 100.2 | 122,762 | 571 | 2,985 |

Spend and order value are in the dataset's currency (GBP).

What the segments mean in practice:

- **Core customers** are recent, regular buyers and make up the majority of the base.
- **At risk** customers last ordered about a year before the data ends and have placed only about two orders. This is nearly half of all customers.
- **High value** is a group of just 27 accounts with around 100 orders each. They behave like wholesale buyers, not typical consumers.

Rough revenue picture, estimated from segment averages multiplied by segment sizes (treat as approximate): Core customers contribute roughly two thirds of spend, the 27 high-value accounts roughly a quarter, and At risk customers roughly a tenth.

### 4.2 Recommendation quality

Evaluated on a 90-day holdout the models never saw. Of the 5,244 customers who received recommendations, 2,288 bought something in the holdout window, so only those could possibly score a hit.

| Method | Precision@3, all customers | Precision@3, customers active in holdout |
|---|---|---|
| Popularity baseline (best sellers the customer hasn't bought) | 1.95% | 4.47% |
| Segment propensity engine | 1.80% | 4.15% |
| **Item-item collaborative filtering** | **2.98%** | **6.83%** |

Other item-item metrics: recall@3 of 0.46% and NDCG@3 of 0.032.

Relative to the popularity baseline, item-item is about 53% better in both views. Relative to the segment propensity engine it is about 65% better.

## 5. Why these results

- **Why the segment engine failed to beat popularity.** Segment propensity gives every customer in a segment nearly the same picks, and in this catalogue those picks overlap heavily with the overall best sellers. It adds little beyond popularity. The first version recommended only 51 distinct products across 5,000+ customers, which shows how little personalisation it provided.
- **Why item-item worked better.** It personalises from each customer's own purchase history: if you bought items that other customers bought together with item X, X is a good next offer. That captures product relationships a segment average cannot.
- **Why the absolute numbers are low.** There are thousands of products, each customer buys a tiny fraction of them in 90 days, and only three recommendations are scored. Low absolute precision is normal for this kind of task. The meaningful comparison is the relative lift over the baseline.
- **Why the segments are lopsided.** Wholesale customers place enormous orders, so a handful of accounts form their own cluster. That is a property of the data, not a modelling error, and it is a real business insight.

## 6. Business impact and recommended actions

These are conclusions from the data. They have not been tested in a live campaign.

1. **Use item-item recommendations for offers.** On the same customers and the same period it surfaced noticeably more products customers went on to buy than a best-seller list.
2. **Run a win-back campaign for the At risk segment.** It is about 45% of customers, with long inactivity and low order counts. Even a small reactivation rate on a group this large matters, and this is the clearest targeting opportunity.
3. **Treat the 27 High value accounts as key accounts.** At this size, offers are a sales-relationship decision, not an automated campaign.
4. **Do not assume segmentation alone improves recommendations.** The segment engine did not beat a simple baseline. Always measure against one.

## 7. What went wrong along the way (and how it was fixed)

The project is documented honestly, including the bugs that taught the most:

1. **Parquet save failed.** Audit metadata stored as a DataFrame in `DataFrame.attrs` could not be serialised when writing parquet. Fixed by clearing `attrs` on a copy before saving.
2. **List vs set error in evaluation.** Ground-truth purchases were lists while recommendations were sets, which broke the intersection. Fixed by converting to sets.
3. **Every metric was exactly 0.0.** This was the most important find. The recommender was given the full transaction history, including the holdout window, and excluded items "already bought". That removed every item a customer later bought, so hits were impossible. Fixed by giving the recommender only pre-cutoff history. This is a classic data leakage trap, caught because zero scores on thousands of customers are not credible.
4. **NDCG was a copy of precision.** The first evaluator returned precision under the NDCG name. Replaced with a proper rank-aware NDCG calculation.
5. **The first engine did not beat the baseline.** Rather than reporting it as a success, the popularity comparison exposed it. That led to building the item-item recommender, which does beat it.

## 8. Technology

- **Python 3.14**, **pandas** and **NumPy** for data handling and the similarity computation
- **scikit-learn** for clustering and the propensity model
- **openpyxl** (Excel input) and **pyarrow** (parquet files)
- **Streamlit** for a simple viewer of the generated metrics
- **pytest** for automated tests (6 tests passing)

## 9. How to run

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset into `data/raw/online_retail_II.xlsx`, then:

```
python -m src.pipeline --input data/raw/online_retail_II.xlsx --output artifacts
python scripts\baseline.py
pytest -q
streamlit run src/app.py
```

`scripts\baseline.py` compares the recommendations in `artifacts/` with the popularity baseline. Run it from the project root. Loading the Excel file can take a few minutes.

Outputs are written to `artifacts/`: `customer_table.parquet` and `.csv`, `segment_profiles.csv`, `recommendations.csv`, `metrics.json` and the saved segmentation model in `models/`.

## 10. Project structure

```
data/raw/            dataset goes here (not committed)
src/
  data.py            ingestion, cleaning, audit log
  features.py        cutoff and RFM features
  segment.py         customer segmentation and profiles
  propensity.py      propensity model
  recommend.py       segment propensity recommender
  itemitem.py        item-item collaborative filtering recommender
  evaluate.py        precision, recall, NDCG on the holdout
  reports.py         writes outputs to artifacts/
  pipeline.py        end-to-end run
  app.py             Streamlit viewer
scripts/baseline.py  popularity baseline comparison
tests/               automated tests
```

## 11. Limitations

- **One retailer, one period.** Results come from a single UK gift-ware business between 2009 and 2011 and may not transfer to other retailers.
- **Wholesale skew.** Many customers are businesses, so patterns differ from typical consumer retail.
- **Single holdout window, no significance test.** The 53% lift is based on one 90-day split. It is a large gap on thousands of customers, but I have not run a formal statistical test or repeated the split over several periods.
- **Offline evaluation only.** Recommendations are scored against later purchases. That shows the model predicts behaviour, not that an offer would cause a purchase. Only a live A/B test can show that.
- **Customers without history are not covered.** New customers have no past purchases to base recommendations on (the cold-start problem). About 5,244 of 5,267 segmented customers received recommendations.
- **Product coverage.** The similarity matrix uses the 2,500 most frequently purchased products, so rarer products are never recommended.
- **No price, margin or stock data.** Recommendations optimise for likelihood of purchase, not profit.
- **Segment quality not formally scored.** Silhouette or similar cluster-quality metrics are not reported.
- **Low absolute precision.** See section 5. Compare methods against the baseline, not against 100%.

## 12. Possible next steps

- Blend popularity and item-item scores into a hybrid recommender.
- Repeat the evaluation across several rolling cutoffs and add confidence intervals.
- Add a customer lifetime value estimate per segment and weight offers by expected profit.
- Add a cold-start fallback for customers with no history.
- Design an A/B test to measure real uplift from the recommendations.

## 13. Data source and credit

Chen, D. (2012). Online Retail II dataset, UCI Machine Learning Repository. Licensed under CC BY 4.0.
