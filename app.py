import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Olist Customer Analytics", page_icon="🛒", layout="wide")
st.title("Olist | Customer Analytics & Predictive Intelligence")
st.caption("Big-data project demo · Kafka event capture + PySpark processing are executed in the companion Kaggle cloud pipeline.")

@st.cache_data(show_spinner="Connecting to Kaggle and loading Olist data...")
def load_data():
    # Optional: put KAGGLE_API_TOKEN in Streamlit Cloud > App settings > Secrets.
    try:
        if "KAGGLE_API_TOKEN" in st.secrets:
            os.environ["KAGGLE_API_TOKEN"] = st.secrets["KAGGLE_API_TOKEN"]
    except Exception:
        pass
    root = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    def locate(name):
        found = list(root.rglob(name))
        if not found:
            raise FileNotFoundError(f"Could not find {name} in Kaggle dataset")
        return found[0]

    customers = pd.read_csv(locate("olist_customers_dataset.csv"), usecols=["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"], dtype={"customer_zip_code_prefix": str})
    orders = pd.read_csv(locate("olist_orders_dataset.csv"), parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(locate("olist_order_items_dataset.csv"), usecols=["order_id", "product_id", "price", "freight_value"])
    products = pd.read_csv(locate("olist_products_dataset.csv"), usecols=["product_id", "product_category_name"])
    reviews = pd.read_csv(locate("olist_order_reviews_dataset.csv"), usecols=["order_id", "review_score"])
    trans_path = locate("product_category_name_translation.csv")
    trans = pd.read_csv(trans_path)

    items["line_value"] = items["price"].fillna(0) + items["freight_value"].fillna(0)
    totals = items.groupby("order_id", as_index=False).agg(order_value=("line_value", "sum"))
    orders = orders.merge(customers, on="customer_id", how="left").merge(totals, on="order_id", how="left")
    orders = orders.merge(reviews.groupby("order_id", as_index=False).review_score.mean(), on="order_id", how="left")
    orders["order_value"] = orders["order_value"].fillna(0)
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    orders = orders.dropna(subset=["order_purchase_timestamp", "customer_unique_id"])

    products = products.merge(trans, on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(products["product_category_name"]).fillna("unknown")

    # Aggregate the 1M-row ZIP lookup before joining it to customers.
    geo_path = locate("olist_geolocation_dataset.csv")
    geo_raw = pd.read_csv(geo_path, usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"], dtype={"geolocation_zip_code_prefix": str})
    geo = (geo_raw.groupby("geolocation_zip_code_prefix", as_index=False)
           .agg(latitude=("geolocation_lat", "mean"), longitude=("geolocation_lng", "mean"))
           .rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix"}))
    customer_geo = customers.merge(geo, on="customer_zip_code_prefix", how="left")

    line = (items.merge(orders[["order_id", "customer_unique_id"]], on="order_id", how="inner")
            .merge(products[["product_id", "category"]], on="product_id", how="left"))
    line["category"] = line["category"].fillna("unknown")
    return orders, customers, line, products, customer_geo

@st.cache_resource(show_spinner="Training repeat-purchase model...")
def train_repeat_model(orders):
    d = orders[(orders["order_status"] == "delivered") & orders["order_purchase_timestamp"].notna()].copy()
    max_date = d["order_purchase_timestamp"].max()
    cutoff = max_date - pd.Timedelta(days=180)
    past = d[d["order_purchase_timestamp"] < cutoff]
    future_ids = set(d.loc[d["order_purchase_timestamp"] >= cutoff, "customer_unique_id"].dropna())
    f = (past.groupby("customer_unique_id").agg(
        order_count=("order_id", "count"), total_spend=("order_value", "sum"),
        avg_order_value=("order_value", "mean"), last_purchase=("order_purchase_timestamp", "max"),
        customer_state=("customer_state", lambda x: x.mode().iloc[0] if not x.mode().empty else "UNKNOWN")
    ).reset_index())
    f["recency_days"] = (cutoff - f["last_purchase"]).dt.days
    f["repeat_next_180d"] = f["customer_unique_id"].isin(future_ids).astype(int)
    f = f.drop(columns=["last_purchase"])
    metrics = {"cutoff": cutoff.date().isoformat(), "customers": len(f), "repeat_customers": int(f.repeat_next_180d.sum())}
    if f.repeat_next_180d.nunique() < 2 or f.repeat_next_180d.value_counts().min() < 2:
        return f, None, metrics
    numeric = ["order_count", "total_spend", "avg_order_value", "recency_days"]
    categorical = ["customer_state"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=400, class_weight="balanced"))])
    X, y = f[numeric + categorical], f["repeat_next_180d"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics.update({"precision": precision_score(yte, pred, zero_division=0),
                    "recall": recall_score(yte, pred, zero_division=0),
                    "f1": f1_score(yte, pred, zero_division=0),
                    "roc_auc": roc_auc_score(yte, p)})
    # Refit on all labelled customers for interactive demo predictions.
    pipe.fit(X, y)
    return f, pipe, metrics

try:
    orders, customers, line, products, customer_geo = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.info("The Olist Kaggle dataset is public. If Kaggle requests authentication, create an API token and add KAGGLE_API_TOKEN in Streamlit Cloud → App settings → Secrets.")
    st.stop()

# Common filtered dataset and headline KPIs
valid = orders[orders["order_status"] == "delivered"].copy()
repeat_rate = valid.groupby("customer_unique_id").size().gt(1).mean() * 100
k1, k2, k3, k4 = st.columns(4)
k1.metric("Delivered orders", f"{len(valid):,}")
k2.metric("Unique customers", f"{valid.customer_unique_id.nunique():,}")
k3.metric("Gross order value", f"R$ {valid.order_value.sum():,.0f}")
k4.metric("Repeat-customer share", f"{repeat_rate:.2f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Repeat-Purchase Prediction", "Product Recommendations", "Spatial & Temporal", "Streaming Architecture"])

with tab1:
    st.subheader("Customer and sales overview")
    monthly = (valid.assign(month=valid.order_purchase_timestamp.dt.to_period("M").astype(str))
               .groupby("month", as_index=False).agg(orders=("order_id", "count"), sales=("order_value", "sum")))
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(monthly, x="month", y="orders", markers=True, title="Monthly delivered orders"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(monthly, x="month", y="sales", markers=True, title="Monthly order value (BRL)"), use_container_width=True)
    state = valid.groupby("customer_state", as_index=False).agg(orders=("order_id", "count"), sales=("order_value", "sum")).sort_values("sales", ascending=False).head(15)
    st.plotly_chart(px.bar(state, x="customer_state", y="sales", hover_data=["orders"], title="Top states by order value"), use_container_width=True)
    st.caption("Olist contains historical anonymized marketplace transactions from 2016–2018; the dashboard analyses that historical period, not live customer accounts.")

with tab2:
    st.subheader("Predict whether a customer will purchase again")
    st.write("Target: at least one delivered purchase in the final 180 days of the dataset, using only each customer's earlier purchase history as features.")
    features, model, metrics = train_repeat_model(orders)
    if model is None:
        st.warning("The available repeat-purchase labels do not support a two-class model. Inspect the dataset split before interpreting predictions.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics['precision']:.3f}")
        m2.metric("Recall", f"{metrics['recall']:.3f}")
        m3.metric("F1 score", f"{metrics['f1']:.3f}")
        m4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        ids = features.sort_values("order_count", ascending=False).customer_unique_id.head(300).tolist()
        chosen = st.selectbox("Select a customer (top 300 by historical order count)", ids)
        row = features.loc[features.customer_unique_id == chosen].iloc[0]
        Xone = pd.DataFrame([row[["order_count", "total_spend", "avg_order_value", "recency_days", "customer_state"]]])
        probability = float(model.predict_proba(Xone)[0, 1])
        st.metric("Estimated repeat-purchase probability", f"{probability:.1%}")
        st.caption(f"Customer state: {row.customer_state} · Previous orders: {int(row.order_count)} · Historical spend: R$ {row.total_spend:,.2f} · Prediction cutoff: {metrics['cutoff']} · Horizon: 180 days")
        st.progress(min(max(probability, 0.0), 1.0))
    st.caption("This is a historical holdout demonstration, not a guarantee of future buying behaviour. Metrics are evaluated on a stratified customer-level test split.")

with tab3:
    st.subheader("Personalized product recommendations")
    counts = line.groupby("customer_unique_id").size().sort_values(ascending=False)
    choices = counts.head(300).index.tolist()
    chosen_rec = st.selectbox("Choose a customer", choices, key="rec_customer")
    customer_line = line[line.customer_unique_id == chosen_rec]
    if customer_line.empty:
        st.info("No purchase history found for this customer.")
    else:
        preferred = customer_line.groupby("category").size().sort_values(ascending=False).index[0]
        already = set(customer_line.product_id.dropna())
        candidates = (line[line.category == preferred]
                      .groupby(["product_id", "category"], as_index=False)
                      .agg(popularity=("order_id", "count"), avg_price=("price", "mean"))
                      .query("product_id not in @already")
                      .sort_values("popularity", ascending=False).head(10))
        st.write(f"Most-purchased category for this customer: **{preferred}**")
        if candidates.empty:
            st.info("No unseen products are available in the preferred category; try another customer.")
        else:
            st.dataframe(candidates, use_container_width=True, hide_index=True)
        st.caption("Hybrid recommender: customer-preferred category + globally popular products in that category, excluding previously purchased products.")

with tab4:
    st.subheader("Spatiotemporal customer analytics")
    monthly = (valid.assign(month=valid.order_purchase_timestamp.dt.to_period("M").astype(str))
               .groupby("month", as_index=False).agg(orders=("order_id", "count"), sales=("order_value", "sum")))
    st.plotly_chart(px.area(monthly, x="month", y="sales", title="Order value over time (BRL)"), use_container_width=True)
    city = (customer_geo.dropna(subset=["latitude", "longitude"])
            .groupby(["customer_state", "customer_city"], as_index=False)
            .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), customers=("customer_unique_id", "nunique"))
            .sort_values("customers", ascending=False).head(400))
    if not city.empty:
        fig = px.scatter_geo(city, lat="latitude", lon="longitude", size="customers", color="customer_state",
                             hover_name="customer_city", hover_data={"customers": True}, scope="south america",
                             title="Customer concentration by city (Brazil)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Geolocation coordinates were not available for the customer ZIP prefixes.")

with tab5:
    st.subheader("Kafka → PySpark streaming pipeline")
    st.markdown("""
    **Implemented in the companion Kaggle cloud pipeline:**
    1. Olist orders are prepared with customer IDs, state, timestamp and order value.
    2. A one-node Apache Kafka KRaft broker runs in the Kaggle cloud session.
    3. A Kafka producer publishes timestamped order events to the `olist-orders` topic.
    4. Spark Structured Streaming consumes the Kafka topic directly using the Spark-Kafka connector.
    5. PySpark Structured Streaming parses the Kafka messages and computes monthly sales, state-wise sales and customer aggregates.
    6. PySpark also performs ZIP-prefix geolocation aggregation; the recommendation and repeat-purchase tasks are demonstrated in the project pipeline.
    """)
    st.info("The Kafka broker is intentionally not hosted inside this Streamlit app. Run kaggle_pipeline.py in Kaggle to demonstrate actual Kafka publishing plus Spark Structured Streaming consumption and PySpark processing. This keeps the public dashboard lightweight and avoids a permanently running broker or paid VM.")
    sample = valid.sort_values("order_purchase_timestamp", ascending=False)[["order_id", "customer_unique_id", "customer_state", "order_purchase_timestamp", "order_value"]].head(20)
    st.write("Sample timestamped order events from the same Olist source:")
    st.dataframe(sample, use_container_width=True, hide_index=True)
