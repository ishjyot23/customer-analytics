import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import kagglehub
from modeling import (
    MODEL_FEATURES,
    prepare_repeat_purchase_data,
    fit_repeat_purchase_model,
)

st.set_page_config(page_title="Olist Customer Analytics", page_icon="🛒", layout="wide")
st.title("Olist | Customer Analytics & Predictive Intelligence")
st.caption(
    "Big-data mini project · Historical Olist order events are replayed through Apache Kafka "
    "and processed using PySpark in the companion Kaggle pipeline."
)


@st.cache_data(show_spinner="Connecting to Kaggle and loading Olist data...")
def load_data():
    # Optional: set KAGGLE_API_TOKEN under Streamlit Cloud → App settings → Secrets.
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

    customers = pd.read_csv(
        locate("olist_customers_dataset.csv"),
        usecols=[
            "customer_id", "customer_unique_id", "customer_zip_code_prefix",
            "customer_city", "customer_state",
        ],
        dtype={"customer_zip_code_prefix": str},
    )
    orders = pd.read_csv(locate("olist_orders_dataset.csv"), parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(
        locate("olist_order_items_dataset.csv"),
        usecols=["order_id", "product_id", "price", "freight_value"],
    )
    products = pd.read_csv(
        locate("olist_products_dataset.csv"),
        usecols=["product_id", "product_category_name"],
    )
    reviews = pd.read_csv(
        locate("olist_order_reviews_dataset.csv"),
        usecols=["order_id", "review_score"],
    )
    translation = pd.read_csv(locate("product_category_name_translation.csv"))

    items["line_value"] = items["price"].fillna(0) + items["freight_value"].fillna(0)
    totals = items.groupby("order_id", as_index=False).agg(order_value=("line_value", "sum"))
    orders = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(totals, on="order_id", how="left")
        .merge(reviews.groupby("order_id", as_index=False).review_score.mean(), on="order_id", how="left")
    )
    orders["order_value"] = orders["order_value"].fillna(0)
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    orders = orders.dropna(subset=["order_purchase_timestamp", "customer_unique_id"])

    products = products.merge(translation, on="product_category_name", how="left")
    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )

    # Aggregate the roughly one-million-row ZIP lookup before joining to customers.
    geo_raw = pd.read_csv(
        locate("olist_geolocation_dataset.csv"),
        usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
        dtype={"geolocation_zip_code_prefix": str},
    )
    geo = (
        geo_raw.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(latitude=("geolocation_lat", "mean"), longitude=("geolocation_lng", "mean"))
        .rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix"})
    )
    customer_geo = customers.merge(geo, on="customer_zip_code_prefix", how="left")

    line = (
        items.merge(orders[["order_id", "customer_unique_id"]], on="order_id", how="inner")
        .merge(products[["product_id", "category"]], on="product_id", how="left")
    )
    line["category"] = line["category"].fillna("unknown")
    return orders, customers, line, products, customer_geo


@st.cache_data(show_spinner="Preparing customer features and inspecting target classes...")
def prepare_repeat_model_data(orders):
    return prepare_repeat_purchase_data(orders, horizon_days=180)


@st.cache_resource(show_spinner="Training and evaluating the repeat-purchase model...")
def fit_repeat_model(features, target_info):
    return fit_repeat_purchase_model(features, target_info)


try:
    orders, customers, line, products, customer_geo = load_data()
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.info(
        "The Olist Kaggle dataset is public. If Kaggle requests authentication, create a Kaggle API token "
        "and add KAGGLE_API_TOKEN in Streamlit Cloud → App settings → Secrets."
    )
    st.stop()

valid = orders[orders["order_status"] == "delivered"].copy()
repeat_rate = valid.groupby("customer_unique_id").size().gt(1).mean() * 100
k1, k2, k3, k4 = st.columns(4)
k1.metric("Delivered orders", f"{len(valid):,}")
k2.metric("Unique customers", f"{valid.customer_unique_id.nunique():,}")
k3.metric("Gross order value", f"R$ {valid.order_value.sum():,.0f}")
k4.metric("Repeat-customer share", f"{repeat_rate:.2f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Repeat-Purchase Prediction", "Product Recommendations", "Spatial & Temporal", "Streaming Architecture"]
)

with tab1:
    st.subheader("Customer and sales overview")
    monthly = (
        valid.assign(month=valid.order_purchase_timestamp.dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(orders=("order_id", "count"), sales=("order_value", "sum"))
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.line(monthly, x="month", y="orders", markers=True, title="Monthly delivered orders"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.line(monthly, x="month", y="sales", markers=True, title="Monthly order value (BRL)"),
            use_container_width=True,
        )
    state = (
        valid.groupby("customer_state", as_index=False)
        .agg(orders=("order_id", "count"), sales=("order_value", "sum"))
        .sort_values("sales", ascending=False)
        .head(15)
    )
    st.plotly_chart(
        px.bar(state, x="customer_state", y="sales", hover_data=["orders"], title="Top states by order value"),
        use_container_width=True,
    )
    st.caption(
        "Olist contains historical anonymized marketplace transactions from 2016–2018; "
        "the dashboard analyses that historical period, not live customer accounts."
    )

with tab2:
    st.subheader("Predict whether a customer will purchase again")
    st.write(
        "Target: at least one delivered purchase in the final 180 days of the dataset, "
        "using only each customer's earlier delivered-purchase history as model features."
    )
    features, target_info = prepare_repeat_model_data(orders)

    st.markdown("#### Target class distribution — inspected before model training")
    class_table = pd.DataFrame(target_info.get("class_distribution", []))
    if not class_table.empty:
        st.dataframe(class_table, use_container_width=True, hide_index=True)
    st.caption(
        f"Prediction cutoff: {target_info.get('cutoff', 'N/A')} · "
        f"Future observation window: {target_info.get('horizon_days', 180)} days · "
        f"Positive repeat-purchase rate: {target_info.get('positive_rate_pct', 0):.3f}%"
    )

    # The fitting step is deliberately called only after displaying the actual class counts.
    model, metrics = fit_repeat_model(features, target_info)
    if model is None:
        st.warning(metrics.get("warning", "Insufficient class support to train a reliable model."))
    else:
        st.markdown("#### Held-out test-set performance")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Precision", f"{metrics['precision']:.3f}")
        metric_cols[1].metric("Recall", f"{metrics['recall']:.3f}")
        metric_cols[2].metric("F1 score", f"{metrics['f1']:.3f}")
        metric_cols[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        metric_cols[4].metric("PR-AUC / Avg. precision", f"{metrics['average_precision']:.3f}")
        st.caption(
            f"Decision threshold: {metrics['decision_threshold']:.4f} (selected on validation data to maximize F1). "
            f"Average-precision baseline (test positive rate): {metrics['average_precision_baseline']:.3f}; "
            f"model AP lift over baseline: {metrics['average_precision_lift_over_prevalence']:.2f}×. "
            f"Always-negative baseline accuracy: {metrics['always_negative_baseline_accuracy']:.3f}. "
            f"Probability calibration used: {'Yes' if metrics['probability_calibrated'] else 'No'}."
        )
        if metrics['average_precision_lift_over_prevalence'] <= 1.0:
            st.warning(
                "The model's average precision is not above the test-set prevalence baseline. "
                "Treat its predictions as experimental; the selected features have not demonstrated useful ranking skill yet."
            )
        with st.expander("Train / validation / test class distribution"):
            split_rows = []
            for split_name, rows in metrics.get("split_distribution", {}).items():
                for row in rows:
                    split_rows.append({"split": split_name, **row})
            if split_rows:
                st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
        with st.expander("Test confusion matrix and model details"):
            st.json(metrics.get("test_confusion_matrix", {}))
            st.write("Class-imbalance strategy:", metrics.get("class_imbalance_handling"))

        st.markdown("#### Customer-level prediction demo")
        ids = features.sort_values("order_count", ascending=False).customer_unique_id.head(300).tolist()
        if ids:
            chosen = st.selectbox("Select a customer (top 300 by historical order count)", ids)
            row = features.loc[features.customer_unique_id == chosen].iloc[0]
            one_customer = pd.DataFrame([{column: row[column] for column in MODEL_FEATURES}])
            score = float(model.predict_proba(one_customer)[0, 1])
            if metrics.get("probability_calibrated"):
                st.metric("Estimated repeat-purchase likelihood", f"{score:.1%}")
            else:
                st.metric("Repeat-purchase model score (uncalibrated)", f"{score:.1%}")
            predicted = score >= metrics["decision_threshold"]
            st.write(
                "Model classification:",
                "Repeat-purchase candidate" if predicted else "No repeat purchase predicted at the selected threshold",
            )
            st.caption(
                f"Customer state: {row.customer_state} · Previous orders: {int(row.order_count)} · "
                f"Historical spend: R$ {row.total_spend:,.2f} · "
                f"Cutoff: {metrics['cutoff']} · Horizon: 180 days"
            )
            st.progress(min(max(score, 0.0), 1.0))
    st.caption(
        "Evaluation uses a stratified customer-level train/validation/test split. "
        "The decision threshold is selected on validation data; the final metrics are calculated on an untouched test set. "
        "The model is a historical demonstration, not a guarantee of future customer behaviour."
    )

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
        candidates = (
            line[line.category == preferred]
            .groupby(["product_id", "category"], as_index=False)
            .agg(popularity=("order_id", "count"), avg_price=("price", "mean"))
            .query("product_id not in @already")
            .sort_values("popularity", ascending=False)
            .head(10)
        )
        st.write(f"Most-purchased category for this customer: **{preferred}**")
        if candidates.empty:
            st.info("No unseen products are available in the preferred category; try another customer.")
        else:
            st.dataframe(candidates, use_container_width=True, hide_index=True)
        st.caption(
            "Baseline recommender: customer-preferred category + globally popular products in that category, "
            "excluding products already purchased by that customer."
        )

with tab4:
    st.subheader("Spatiotemporal customer analytics")
    monthly = (
        valid.assign(month=valid.order_purchase_timestamp.dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(orders=("order_id", "count"), sales=("order_value", "sum"))
    )
    st.plotly_chart(
        px.area(monthly, x="month", y="sales", title="Order value over time (BRL)"),
        use_container_width=True,
    )

    st.markdown("#### Spatiotemporal heatmap — customer state × purchase month")
    heat_metric = st.radio(
        "Heatmap measure",
        ["Sales (BRL)", "Delivered orders"],
        horizontal=True,
        key="state_month_measure",
    )
    metric_column = "order_value" if heat_metric == "Sales (BRL)" else "order_id"
    aggregation = "sum" if heat_metric == "Sales (BRL)" else "count"
    state_month = (
        valid.assign(month=valid.order_purchase_timestamp.dt.to_period("M").astype(str))
        .groupby(["customer_state", "month"], as_index=False)
        .agg(value=(metric_column, aggregation))
    )
    heat_matrix = (
        state_month.pivot(index="customer_state", columns="month", values="value")
        .fillna(0)
    )
    heat_matrix = heat_matrix.loc[heat_matrix.sum(axis=1).sort_values(ascending=False).index]
    heat_title = "Monthly sales by customer state (BRL)" if heat_metric == "Sales (BRL)" else "Monthly delivered orders by customer state"
    heat_fig = px.imshow(
        heat_matrix,
        aspect="auto",
        labels={"x": "Purchase month", "y": "Customer state", "color": heat_metric},
        title=heat_title,
        color_continuous_scale="Blues",
    )
    heat_fig.update_layout(height=650)
    st.plotly_chart(heat_fig, use_container_width=True)
    st.caption(
        "Each cell combines a geographic dimension (customer state) with a temporal dimension (purchase month). "
        "State is derived from the Olist customer record; sales include item price plus freight."
    )

    city = (
        customer_geo.dropna(subset=["latitude", "longitude"])
        .groupby(["customer_state", "customer_city"], as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            customers=("customer_unique_id", "nunique"),
        )
        .sort_values("customers", ascending=False)
        .head(400)
    )
    if not city.empty:
        geo_fig = px.scatter_geo(
            city,
            lat="latitude",
            lon="longitude",
            size="customers",
            color="customer_state",
            hover_name="customer_city",
            hover_data={"customers": True},
            scope="south america",
            title="Customer concentration by city (Brazil)",
        )
        st.plotly_chart(geo_fig, use_container_width=True)
    else:
        st.warning("Geolocation coordinates were not available for the customer ZIP prefixes.")

with tab5:
    st.subheader("Kafka → PySpark streaming pipeline")
    st.markdown(
        """
        **Implemented in the companion Kaggle cloud pipeline:**
        1. Olist orders are prepared with customer IDs, state, timestamp and order value.
        2. A one-node Apache Kafka KRaft broker runs in the Kaggle cloud session.
        3. A Kafka producer publishes timestamped historical order events to a Kafka topic.
        4. Spark Structured Streaming consumes the topic using the Spark-Kafka connector.
        5. PySpark and Spark SQL compute monthly, state-wise and state-by-month metrics.
        6. ZIP-prefix geolocation is aggregated for geographic customer analysis.
        7. Customer repeat-purchase features and category/popularity recommendations are generated.
        """
    )
    st.info(
        "Run kaggle_pipeline.py in Kaggle to demonstrate actual Kafka publishing and Spark Structured Streaming consumption. "
        "The public dashboard is independent of that temporary Kaggle session and does not claim that Kafka is running continuously."
    )
    sample = valid.sort_values("order_purchase_timestamp", ascending=False)[
        ["order_id", "customer_unique_id", "customer_state", "order_purchase_timestamp", "order_value"]
    ].head(20)
    st.write("Sample timestamped order events from the Olist source:")
    st.dataframe(sample, use_container_width=True, hide_index=True)
