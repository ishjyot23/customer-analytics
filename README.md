# Olist Customer Analytics & Predictive Intelligence

A cloud-first Big Data mini project using the Olist Brazilian E-Commerce dataset.

## Components
- Apache Kafka (single-node KRaft broker inside a Kaggle Notebook session) for timestamped order-event capture.
- PySpark for distributed ETL, monthly/state analytics, and ZIP-code geographic aggregation.
- Repeat-purchase propensity model using a chronological 180-day label window.
- Personalized category/popularity-based product recommender.
- Streamlit Community Cloud dashboard, loading the public Olist dataset via KaggleHub.

## Run the Big Data pipeline in Kaggle
1. Open https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and create a new notebook.
2. Ensure the dataset is attached as an Input and Internet is enabled in notebook settings.
3. Add `kaggle_pipeline.py` to the notebook session (or paste its contents into a cell) and run it. It installs `kafka-python` and `pyspark`, downloads Kafka into Kaggle's cloud workspace, runs the broker, publishes Olist order events to Kafka, consumes the topic with Spark Structured Streaming, then analyses the stream with PySpark.
4. Evidence files are written under `/kaggle/working/olist_outputs/`.

## Deploy the dashboard
1. Create a public GitHub repository and add `app.py` and `requirements.txt` at the repository root.
2. Open https://share.streamlit.io, sign in with GitHub, choose the repository and select `app.py` as the entrypoint.
3. Deploy. The app downloads the public Olist dataset inside the Streamlit Cloud runtime; no Olist CSVs need to be committed to GitHub or downloaded to your PC.
4. If Kaggle requests authentication, create a Kaggle API token and add `KAGGLE_API_TOKEN = "..."` under Streamlit Community Cloud → App settings → Secrets. Never commit the token to GitHub.

## Dataset
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
License: CC BY-NC-SA 4.0. Attribute the dataset and respect its non-commercial/share-alike terms.

## Interpretation notes
- Olist is historical anonymized data (2016–2018), not a live feed. Kafka streaming is demonstrated by replaying timestamped historical order events.
- The repeat-purchase model uses earlier delivered orders as features and the final 180 days as the future label window.
- The recommender recommends unseen products from the customer's most-purchased category, ranked by dataset-wide popularity within that category.
- Kaggle's notebook session is temporary. The Streamlit dashboard is deployed independently and does not require a continuously running Kafka broker.
