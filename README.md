# Olist Customer Analytics & Predictive Intelligence

Cloud-first Big Data mini project built on the public [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## What the project demonstrates

- **Apache Kafka:** one-node KRaft broker in a Kaggle Notebook cloud session; historical order events are replayed to a Kafka topic.
- **Spark Structured Streaming:** consumes Kafka records directly using Spark's Kafka source.
- **PySpark + Spark SQL:** event cleaning, monthly and state-wise metrics, and state-by-month aggregations.
- **Predictive analytics:** repeat-purchase classification. A customer is positive when they make at least one delivered purchase during the final 180-day observation window, after having at least one earlier delivered order.
- **Class-imbalance handling:** prints target distribution before training, avoids `class_weight='balanced'` probability distortion, selects a decision threshold on a validation split, uses sigmoid probability calibration when class support allows, and reports precision, recall, F1, ROC-AUC and average precision on a separate test split.
- **Recommendation system:** baseline personalized recommendations using the customer's most-purchased category and popular unseen products within that category.
- **Spatiotemporal analytics:** customer ZIP-prefix geolocation and state-by-month sales/order heatmap.
- **Streamlit dashboard:** loads Olist independently through KaggleHub and visualizes the analysis. The dashboard does not require the Kaggle Kafka broker to stay online.

## Repository files

```text
customer-analytics/
├── app.py                  # Streamlit dashboard
├── modeling.py             # Shared repeat-purchase features/model/evaluation
├── kaggle_pipeline.py      # Kafka + Spark Structured Streaming + PySpark pipeline
├── requirements.txt        # Streamlit Cloud dependencies
├── README.md
└── .gitignore
```

## A. Run the Big Data pipeline in Kaggle

1. Open the Olist dataset page: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Create a Kaggle Notebook and attach the dataset using **Add Input**.
3. Turn **Internet** on in notebook Settings (needed to fetch Kafka and the Spark Kafka connector).
4. Make sure the GitHub repository contains `kaggle_pipeline.py` and `modeling.py` at its root.
5. Run this in a Kaggle code cell, replacing the URL if your repository differs:

   ```python
   !git clone https://github.com/ishjyot23/customer-analytics.git
   !python customer-analytics/kaggle_pipeline.py
   ```

   If the repository is already present, update it instead of cloning again:

   ```python
   !git -C customer-analytics pull --ff-only
   !python customer-analytics/kaggle_pipeline.py
   ```

6. The script installs `pyspark==4.0.0` (without the Spark Connect extra), downloads Kafka 3.7.2 into Kaggle's working directory, formats KRaft storage using the Kafka 3.7-compatible command, starts the broker, creates a topic, replays timestamp-sorted Olist events, and consumes them with Spark Structured Streaming.
7. Successful execution prints producer and consumer event counts. They must match. It then prints PySpark monthly/state/state-by-month analytics, spatial aggregation, class distribution, model metrics, and recommendation examples.
8. Evidence is saved to `/kaggle/working/olist_outputs/`:
   - `kafka_summary.json`
   - `monthly_metrics.csv`
   - `state_metrics.csv`
   - `state_month_metrics.csv`
   - `geo_metrics.csv`
   - `class_distribution.csv`
   - `model_metrics.json`
   - `recommendations_sample.csv`

**Important:** Olist is historical data (2016–2018). Kafka streaming is demonstrated by replaying those timestamped historical records as events; it is not a live feed from Olist. Kaggle is session-based, so the broker exists only while the notebook process/session is running.

## B. Deploy the Streamlit dashboard

1. Create a GitHub repository, for example `olist-customer-analytics`.
2. Upload `app.py`, `modeling.py`, `requirements.txt`, `kaggle_pipeline.py`, `README.md`, and `.gitignore` to the repository root. Do not upload raw Olist CSVs or API tokens.
3. Open https://share.streamlit.io and sign in with GitHub.
4. Create an app selecting your repository, branch `main`, and main file `app.py`.
5. Deploy. The dashboard downloads the public Olist dataset within Streamlit's cloud runtime using KaggleHub.
6. If Kaggle authentication is required, add your Kaggle API token in Streamlit Cloud → App settings → Secrets. Never commit it to GitHub.

The dashboard is independent of the Kaggle pipeline. It reloads/analyses the public dataset in Streamlit Cloud and does not claim to display live Kafka offsets or results from a stopped Kaggle session.

## Modelling definition and interpretation

- Only delivered orders are used for the repeat-purchase label.
- The cutoff is 180 days before the last delivered purchase date in the dataset.
- Feature rows are customers with at least one delivered order before the cutoff.
- Features use only pre-cutoff purchases: order count, spend, average order value, recency, active days, active months and most-common state.
- The positive class means the customer made at least one delivered purchase during the final 180 days.
- The target class distribution is printed and saved before fitting. If either class has fewer than 15 examples, model training is skipped rather than presenting unstable metrics.
- Data is split into train, validation and test groups by customer. The validation set selects the F1 decision threshold. The test set is kept out of threshold selection and model fitting.
- Logistic regression is not class-weighted; this avoids artificially inflating the minority class's predicted probabilities. Sigmoid calibration is used when supported by the training class counts.
- Average precision (AP, a PR-AUC summary), precision, recall, F1 and ROC-AUC are reported. AP is compared with the test-set positive-class prevalence; accuracy is accompanied by an always-negative baseline. If AP does not beat prevalence, the dashboard flags the model as experimental rather than implying useful predictive skill.
- The final displayed score is a model-estimated likelihood, not a guarantee of future purchase.

## Recommendation method

For a selected customer, the dashboard finds their most-purchased product category, excludes products already purchased, and recommends popular unseen products within that category. This is an interpretable baseline recommender, not a deep-learning or collaborative-filtering model.

## Dataset attribution and licence

Dataset: Olist Brazilian E-Commerce Public Dataset. Respect the dataset's CC BY-NC-SA 4.0 licence and attribution/non-commercial/share-alike conditions: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
