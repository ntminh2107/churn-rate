"""Interactive Streamlit demo for the customer-churn pipeline."""

from __future__ import annotations

import secrets

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.features import audit_risk_rules
from pipeline.inference import (
    customer_risk_signals,
    generate_random_customer,
    predict_customer,
    train_demo_models,
)


st.set_page_config(
    page_title="ChurnScope — Customer Risk Demo",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    :root {
        --ink: #1f2a44;
        --muted: #6b7280;
        --purple: #6c5ce7;
        --purple-dark: #5145cd;
        --cyan: #00b8d9;
        --green: #10b981;
        --red: #ef4444;
        --amber: #f59e0b;
        --card: rgba(255, 255, 255, 0.94);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 8% 5%, rgba(108, 92, 231, .10), transparent 24rem),
            radial-gradient(circle at 94% 18%, rgba(0, 184, 217, .09), transparent 22rem),
            #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171b36 0%, #25254c 100%);
    }

    .side-brand,
    .side-brand *,
    .sidebar-heading {
        color: #f8faff !important;
    }

    .st-key-model_selector_control [data-testid="stWidgetLabel"] *,
    .st-key-threshold_control [data-testid="stWidgetLabel"] *,
    .st-key-random_seed_control [data-testid="stWidgetLabel"] * {
        color: #f8faff !important;
    }

    .st-key-model_selector_control [data-testid="stSelectbox"]
    div[data-baseweb="select"] > div,
    .st-key-random_seed_control [data-testid="stNumberInput"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background: #ffffff !important;
        border-color: rgba(17, 24, 39, .16) !important;
    }

    .st-key-model_selector_control [data-testid="stSelectbox"]
    div[data-baseweb="select"] *,
    .st-key-model_selector_control [data-testid="stSelectbox"]
    div[role="combobox"],
    .st-key-model_selector_control [data-testid="stSelectbox"] input,
    .st-key-random_seed_control [data-testid="stNumberInput"] input,
    .st-key-random_seed_control [data-testid="stNumberInput"] button {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    .st-key-model_selector_control [data-testid="stSelectbox"] svg,
    .st-key-random_seed_control [data-testid="stNumberInput"] svg {
        fill: #111827 !important;
        color: #111827 !important;
    }

    div[data-baseweb="popover"] [role="listbox"] {
        color: #111827 !important;
        background: #ffffff !important;
    }

    div[data-baseweb="menu"],
    div[data-baseweb="menu"] *,
    ul[role="listbox"],
    ul[role="listbox"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] [role="option"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background: #ede9fe !important;
    }

    div[data-baseweb="popover"] [role="option"]:hover {
        background: #f5f3ff !important;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        border-radius: 24px;
        color: white;
        background:
            linear-gradient(120deg, rgba(20, 24, 58, .98), rgba(81, 69, 205, .94)),
            #25254c;
        box-shadow: 0 20px 55px rgba(52, 47, 120, .22);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        right: -70px;
        top: -120px;
        background: radial-gradient(circle, rgba(0, 220, 255, .34), transparent 68%);
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        padding: .35rem .75rem;
        border: 1px solid rgba(255, 255, 255, .25);
        border-radius: 999px;
        background: rgba(255, 255, 255, .08);
        font-size: .78rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: .8rem 0 .35rem;
        font-size: clamp(2rem, 4vw, 3.35rem);
        line-height: 1.05;
        letter-spacing: -.04em;
    }

    .hero p {
        max-width: 790px;
        margin: 0;
        color: rgba(255, 255, 255, .78);
        font-size: 1rem;
    }

    .side-brand {
        padding: .75rem 0 1.1rem;
        color: #f8faff !important;
    }

    .side-brand strong {
        color: #f8faff !important;
        font-size: 1.35rem;
        letter-spacing: -.03em;
    }

    .side-brand span {
        color: rgba(248, 250, 255, .66) !important;
    }

    .sidebar-heading {
        margin: .15rem 0 .7rem;
        font-size: 1.05rem;
        font-weight: 750;
    }

    .side-status {
        margin: .65rem 0 1.25rem;
        padding: .7rem .8rem;
        border: 1px solid rgba(75, 222, 154, .28);
        border-radius: 12px;
        color: #b9f7d5 !important;
        background: rgba(16, 185, 129, .10);
        font-size: .83rem;
    }

    .section-title {
        margin: .4rem 0 .8rem;
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 750;
        letter-spacing: -.02em;
    }

    .result-card {
        min-height: 150px;
        padding: 1.25rem 1.35rem;
        border: 1px solid rgba(31, 42, 68, .08);
        border-radius: 18px;
        background: var(--card);
        box-shadow: 0 10px 28px rgba(31, 42, 68, .07);
    }

    .result-card .eyebrow {
        color: var(--muted);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .result-card .value {
        margin: .45rem 0 .15rem;
        color: var(--ink);
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }

    .result-card .caption {
        color: var(--muted);
        font-size: .86rem;
    }

    .risk-banner {
        padding: 1.05rem 1.2rem;
        margin: .5rem 0 1rem;
        border-radius: 16px;
        border-left: 5px solid;
        font-weight: 650;
    }

    .risk-high {
        color: #9f1239;
        border-color: #f43f5e;
        background: #fff1f2;
    }

    .risk-low {
        color: #047857;
        border-color: #10b981;
        background: #ecfdf5;
    }

    .signal-chip {
        display: inline-block;
        margin: .2rem .28rem .2rem 0;
        padding: .38rem .62rem;
        border: 1px solid #e8e5ff;
        border-radius: 999px;
        color: #5145cd;
        background: #f5f3ff;
        font-size: .79rem;
        font-weight: 650;
    }

    div[data-testid="stMetric"] {
        padding: .8rem 1rem;
        border: 1px solid rgba(31, 42, 68, .07);
        border-radius: 15px;
        background: rgba(255, 255, 255, .88);
        box-shadow: 0 7px 20px rgba(31, 42, 68, .05);
    }

    div.stButton > button {
        min-height: 2.7rem;
        border: 0;
        border-radius: 12px;
        color: white;
        background: linear-gradient(90deg, var(--purple), var(--purple-dark));
        box-shadow: 0 8px 18px rgba(108, 92, 231, .24);
        font-weight: 700;
    }

    div.stButton > button:hover {
        color: white;
        border: 0;
        transform: translateY(-1px);
        box-shadow: 0 11px 22px rgba(108, 92, 231, .30);
    }

    div[data-baseweb="tab-list"] {
        gap: .35rem;
        padding: .25rem;
        border-radius: 13px;
        background: rgba(255, 255, 255, .76);
    }

    div[data-baseweb="tab"] {
        border-radius: 10px;
        padding: .65rem 1rem;
    }

    .small-note {
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.5;
    }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_demo_bundle():
    return train_demo_models()


@st.cache_data(show_spinner=False)
def make_column_profile(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in frame.columns:
        examples = frame[column].dropna().head(3).tolist()
        records.append(
            {
                "Column": column,
                "Type": str(frame[column].dtype),
                "Missing rate": frame[column].isna().mean(),
                "Unique": frame[column].nunique(dropna=True),
                "Examples": str(examples),
            }
        )
    return pd.DataFrame(records)


def display_result_card(
    eyebrow: str,
    value: str,
    caption: str,
) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="eyebrow">{eyebrow}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_customer_editor(
    random_customer: pd.DataFrame,
    widget_version: int,
) -> pd.DataFrame:
    row = random_customer.iloc[0]
    key = f"customer_{widget_version}"

    st.markdown(
        '<div class="section-title">Thông tin khách hàng thử nghiệm</div>',
        unsafe_allow_html=True,
    )
    first, second, third, fourth = st.columns(4)
    with first:
        credit_score = st.slider(
            "Credit score",
            min_value=300,
            max_value=900,
            value=int(row["CreditScore"]),
            key=f"{key}_credit",
        )
        age = st.slider(
            "Tuổi",
            min_value=18,
            max_value=100,
            value=int(row["Age"]),
            key=f"{key}_age",
        )
        geography_options = ["France", "Germany", "Spain"]
        geography = st.selectbox(
            "Quốc gia",
            geography_options,
            index=geography_options.index(str(row["Geography"])),
            key=f"{key}_geography",
        )
        gender_options = ["Female", "Male"]
        gender = st.selectbox(
            "Giới tính",
            gender_options,
            index=gender_options.index(str(row["Gender"])),
            key=f"{key}_gender",
        )

    with second:
        tenure = st.slider(
            "Tenure (năm)",
            min_value=0,
            max_value=10,
            value=int(row["Tenure"]),
            key=f"{key}_tenure",
        )
        products = st.slider(
            "Số sản phẩm",
            min_value=1,
            max_value=4,
            value=int(row["NumOfProducts"]),
            key=f"{key}_products",
        )
        has_card_label = st.selectbox(
            "Credit card",
            ["Có", "Không"],
            index=0 if int(row["HasCrCard"]) == 1 else 1,
            key=f"{key}_card",
        )
        active_label = st.selectbox(
            "Trạng thái hoạt động",
            ["Đang hoạt động", "Không hoạt động"],
            index=0 if int(row["IsActiveMember"]) == 1 else 1,
            key=f"{key}_active",
        )

    with third:
        balance = st.number_input(
            "Số dư",
            min_value=0.0,
            max_value=300_000.0,
            value=float(row["Balance"]),
            step=1_000.0,
            format="%.2f",
            key=f"{key}_balance",
        )
        salary = st.number_input(
            "Thu nhập ước tính",
            min_value=0.0,
            max_value=250_000.0,
            value=float(row["EstimatedSalary"]),
            step=1_000.0,
            format="%.2f",
            key=f"{key}_salary",
        )
        satisfaction = st.slider(
            "Satisfaction score",
            min_value=1,
            max_value=5,
            value=int(row["Satisfaction Score"]),
            key=f"{key}_satisfaction",
        )

    with fourth:
        card_options = ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]
        card_type = st.selectbox(
            "Card type",
            card_options,
            index=card_options.index(str(row["Card Type"])),
            key=f"{key}_card_type",
        )
        points = st.slider(
            "Point earned",
            min_value=0,
            max_value=1_200,
            value=int(row["Point Earned"]),
            key=f"{key}_points",
        )
        st.markdown(
            """
            <div class="small-note" style="margin-top: 1.9rem">
            Các trường định danh và <code>Complain</code> không xuất hiện vì
            chúng đã bị loại khỏi model pipeline.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": products,
                "HasCrCard": 1 if has_card_label == "Có" else 0,
                "IsActiveMember": (
                    1 if active_label == "Đang hoạt động" else 0
                ),
                "EstimatedSalary": salary,
                "Satisfaction Score": satisfaction,
                "Card Type": card_type,
                "Point Earned": points,
            }
        ]
    )


def render_eda(bundle) -> None:
    eda_frame = bundle.eda_frame.copy()
    eda_option = st.selectbox(
        "Chọn nội dung EDA",
        [
            "Tổng quan dataset",
            "Phân phối churn",
            "Phân phối numerical",
            "Phân tích categorical",
            "Correlation heatmap",
            "Risk-rule audit",
        ],
    )

    if eda_option == "Tổng quan dataset":
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Customers", f"{len(eda_frame):,}")
        metric_2.metric("Model features", len(bundle.data.model_columns))
        metric_3.metric(
            "Churn rate",
            f"{eda_frame[bundle.config.target].mean():.2%}",
        )
        metric_4.metric(
            "Missing cells",
            f"{int(eda_frame.isna().sum().sum()):,}",
        )
        st.markdown("#### Column profile")
        profile = make_column_profile(eda_frame)
        st.dataframe(
            profile,
            width="stretch",
            hide_index=True,
            column_config={
                "Missing rate": st.column_config.NumberColumn(
                    format="percent"
                ),
            },
        )

    elif eda_option == "Phân phối churn":
        label_frame = eda_frame.copy()
        label_frame["Status"] = label_frame[
            bundle.config.target
        ].map({0: "Retained", 1: "Churn"})
        left, right = st.columns([0.9, 1.35])
        with left:
            counts = (
                label_frame["Status"]
                .value_counts()
                .rename_axis("Status")
                .reset_index(name="Customers")
            )
            figure = px.pie(
                counts,
                names="Status",
                values="Customers",
                hole=0.64,
                color="Status",
                color_discrete_map={
                    "Retained": "#6C5CE7",
                    "Churn": "#EF476F",
                },
                title="Target distribution",
            )
            figure.update_traces(textposition="outside", textinfo="percent+label")
            figure.update_layout(
                margin=dict(l=20, r=20, t=60, b=20),
                legend_orientation="h",
            )
            st.plotly_chart(figure, width="stretch")
        with right:
            geo_summary = (
                label_frame.groupby("Geography", as_index=False)
                .agg(
                    Customers=("Status", "size"),
                    Churn_rate=(bundle.config.target, "mean"),
                )
                .sort_values("Churn_rate", ascending=False)
            )
            figure = px.bar(
                geo_summary,
                x="Geography",
                y="Churn_rate",
                text_auto=".1%",
                color="Churn_rate",
                color_continuous_scale=["#DAD7FE", "#6C5CE7", "#EF476F"],
                title="Churn rate theo Geography",
            )
            figure.update_yaxes(tickformat=".0%")
            figure.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(figure, width="stretch")

    elif eda_option == "Phân phối numerical":
        numeric_columns = [
            column
            for column in eda_frame.select_dtypes(include=np.number).columns
            if column != bundle.config.target
        ]
        selected_column = st.selectbox(
            "Chọn numerical feature",
            numeric_columns,
        )
        plot_frame = eda_frame.copy()
        plot_frame["Status"] = plot_frame[
            bundle.config.target
        ].map({0: "Retained", 1: "Churn"})
        figure = px.histogram(
            plot_frame,
            x=selected_column,
            color="Status",
            nbins=35,
            marginal="box",
            barmode="overlay",
            opacity=0.62,
            color_discrete_map={
                "Retained": "#6C5CE7",
                "Churn": "#EF476F",
            },
            title=f"Phân phối {selected_column} theo actual class",
        )
        figure.update_layout(
            bargap=0.04,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(figure, width="stretch")

    elif eda_option == "Phân tích categorical":
        categorical_columns = [
            column
            for column in [
                "Geography",
                "Gender",
                "Card Type",
                "HasCrCard",
                "IsActiveMember",
                "NumOfProducts",
                "Satisfaction Score",
            ]
            if column in eda_frame.columns
        ]
        selected_column = st.selectbox(
            "Chọn categorical feature",
            categorical_columns,
        )
        summary = (
            eda_frame.groupby(selected_column, as_index=False)
            .agg(
                Customers=(bundle.config.target, "size"),
                Churn_rate=(bundle.config.target, "mean"),
            )
            .sort_values("Churn_rate", ascending=False)
        )
        left, right = st.columns(2)
        with left:
            figure = px.bar(
                summary,
                x=selected_column,
                y="Customers",
                text_auto=True,
                color="Customers",
                color_continuous_scale=["#DAD7FE", "#6C5CE7"],
                title=f"Customer count theo {selected_column}",
            )
            figure.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure, width="stretch")
        with right:
            figure = px.bar(
                summary,
                x=selected_column,
                y="Churn_rate",
                text_auto=".1%",
                color="Churn_rate",
                color_continuous_scale=["#D1FAE5", "#F59E0B", "#EF476F"],
                title=f"Churn rate theo {selected_column}",
            )
            figure.update_yaxes(tickformat=".0%")
            figure.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure, width="stretch")

    elif eda_option == "Correlation heatmap":
        correlation = eda_frame.select_dtypes(include=np.number).corr()
        figure = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson correlation — numerical features",
        )
        figure.update_layout(
            height=720,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(figure, width="stretch")

    else:
        audit = audit_risk_rules(
            bundle.data.X_train,
            bundle.data.y_train,
        )
        st.caption(
            "Các rule được audit trên train split, không sử dụng external test."
        )
        st.dataframe(
            audit.style.format(
                {
                    "Support": "{:,}",
                    "Exited": "{:,}",
                    "Non-exited": "{:,}",
                    "Exit rate": "{:.2%}",
                    "Lift vs train baseline": "{:.2f}x",
                }
            ).background_gradient(
                subset=["Exit rate", "Lift vs train baseline"],
                cmap="YlOrRd",
            ),
            width="stretch",
            hide_index=True,
        )


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">◈ ML demo · frozen tuned models</div>
        <h1>ChurnScope</h1>
        <p>
            Sinh khách hàng synthetic, tinh chỉnh thông tin và so sánh xác suất
            churn từ XGBoost, LightGBM và Random Forest trong cùng một trải nghiệm.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Đang khởi tạo feature pipeline và ba tuned models…"):
    demo_bundle = load_demo_bundle()


with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <strong>◈ ChurnScope</strong><br/>
            <span style="opacity:.66;font-size:.8rem">Customer risk laboratory</span>
        </div>
        <div class="side-status">● Model bundle sẵn sàng · 3 models</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-heading">Cấu hình dự đoán</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="model_selector_control"):
        selected_model_name = st.selectbox(
            "Model được highlight",
            list(demo_bundle.artifacts),
            index=0,
            help="Cả ba model vẫn được chạy để tạo bảng so sánh.",
        )
    with st.container(key="threshold_control"):
        operating_threshold = st.slider(
            "Decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(demo_bundle.config.operating_threshold),
            step=0.01,
            help="Probability ≥ threshold sẽ được báo cáo là churn.",
        )

    if "random_seed_value" not in st.session_state:
        st.session_state.random_seed_value = 42
    if "seed_widget_version" not in st.session_state:
        st.session_state.seed_widget_version = 0
    if "customer_version" not in st.session_state:
        st.session_state.customer_version = 0
    if "random_customer" not in st.session_state:
        st.session_state.random_customer = generate_random_customer(
            demo_bundle,
            seed=st.session_state.random_seed_value,
        )

    with st.container(key="random_seed_control"):
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            value=st.session_state.random_seed_value,
            step=1,
            key=f"random_seed_{st.session_state.seed_widget_version}",
        )
    if st.button(
        "🎲 Tạo khách hàng ngẫu nhiên",
        width="stretch",
        type="primary",
    ):
        st.session_state.random_customer = generate_random_customer(
            demo_bundle,
            seed=int(random_seed),
        )
        st.session_state.customer_version += 1

    if st.button("↻ Dùng seed mới", width="stretch"):
        new_seed = secrets.randbelow(1_000_001)
        st.session_state.random_seed_value = new_seed
        st.session_state.seed_widget_version += 1
        st.session_state.random_customer = generate_random_customer(
            demo_bundle,
            seed=new_seed,
        )
        st.session_state.customer_version += 1
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div class="small-note" style="color:rgba(255,255,255,.62)">
        Demo sử dụng split và frozen best parameters từ notebook.
        External test không được dùng để fit models.
        </div>
        """,
        unsafe_allow_html=True,
    )


predict_tab, compare_tab, eda_tab, model_tab = st.tabs(
    [
        "🎯 Predict",
        "⚖️ Compare models",
        "🔎 EDA explorer",
        "🧠 Model info",
    ]
)

with predict_tab:
    customer = render_customer_editor(
        st.session_state.random_customer,
        st.session_state.customer_version,
    )
    predictions = predict_customer(
        demo_bundle,
        customer,
        threshold=operating_threshold,
    )
    selected_result = predictions.loc[
        predictions["Model"] == selected_model_name
    ].iloc[0]
    selected_probability = float(selected_result["Churn probability"])
    churn_prediction = selected_result["Prediction"] == "Churn"

    st.markdown("---")
    left, middle, right = st.columns([1.05, 1, 1])
    with left:
        display_result_card(
            "Selected model",
            str(selected_result["Algorithm"]),
            selected_model_name,
        )
    with middle:
        display_result_card(
            "Predicted churn probability",
            f"{selected_probability:.1%}",
            f"Decision threshold: {operating_threshold:.0%}",
        )
    with right:
        display_result_card(
            "Risk classification",
            str(selected_result["Risk level"]),
            str(selected_result["Prediction"]),
        )

    banner_class = "risk-high" if churn_prediction else "risk-low"
    banner_text = (
        "Model khuyến nghị đưa khách hàng vào danh sách chăm sóc churn."
        if churn_prediction
        else "Model đang xếp khách hàng vào nhóm retained."
    )
    st.markdown(
        f'<div class="risk-banner {banner_class}">{banner_text}</div>',
        unsafe_allow_html=True,
    )

    gauge_color = "#EF476F" if churn_prediction else "#10B981"
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=selected_probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn probability"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": gauge_color, "thickness": 0.34},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, operating_threshold * 100],
                        "color": "#D1FAE5",
                    },
                    {
                        "range": [operating_threshold * 100, 100],
                        "color": "#FECDD3",
                    },
                ],
                "threshold": {
                    "line": {"color": "#1F2A44", "width": 4},
                    "thickness": 0.76,
                    "value": operating_threshold * 100,
                },
            },
        )
    )
    gauge.update_layout(
        height=330,
        margin=dict(l=30, r=30, t=70, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1F2A44"},
    )

    gauge_column, signal_column = st.columns([1.05, 1.35])
    with gauge_column:
        st.plotly_chart(gauge, width="stretch")
    with signal_column:
        st.markdown("#### Tín hiệu nổi bật từ dữ liệu")
        signals = customer_risk_signals(customer)
        if signals:
            chips = "".join(
                f'<span class="signal-chip">{signal}</span>'
                for signal in signals
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success("Không phát hiện rule signal nổi bật.")
        st.markdown(
            """
            <div class="small-note" style="margin-top:1rem">
            Các tín hiệu trên dùng để giải thích dữ liệu đầu vào, không phải
            quan hệ nhân quả hay SHAP contribution của riêng dự đoán này.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Xem raw input gửi vào pipeline"):
            st.dataframe(customer, width="stretch", hide_index=True)

with compare_tab:
    st.markdown(
        '<div class="section-title">So sánh trên cùng một khách hàng</div>',
        unsafe_allow_html=True,
    )
    comparison = predictions.copy()
    comparison["Selected"] = comparison["Model"].eq(selected_model_name)
    comparison["Probability (%)"] = (
        comparison["Churn probability"] * 100
    )

    color_map = {
        "Tuned XGBoost": "#6C5CE7",
        "Tuned LightGBM": "#00B8D9",
        "Tuned Random Forest": "#F59E0B",
    }
    comparison_figure = px.bar(
        comparison,
        x="Model",
        y="Churn probability",
        color="Model",
        text_auto=".1%",
        color_discrete_map=color_map,
        title="Predicted churn probability by model",
    )
    comparison_figure.add_hline(
        y=operating_threshold,
        line_dash="dash",
        line_color="#EF476F",
        annotation_text=f"Threshold {operating_threshold:.0%}",
        annotation_position="top left",
    )
    comparison_figure.update_yaxes(
        range=[0, max(1.0, comparison["Churn probability"].max() * 1.2)],
        tickformat=".0%",
    )
    comparison_figure.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=20),
        plot_bgcolor="rgba(255,255,255,.65)",
    )
    st.plotly_chart(comparison_figure, width="stretch")

    st.dataframe(
        comparison[
            [
                "Selected",
                "Model",
                "Probability (%)",
                "Prediction",
                "Risk level",
                "CV PR-AUC",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "Selected": st.column_config.CheckboxColumn(disabled=True),
            "Probability (%)": st.column_config.ProgressColumn(
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "CV PR-AUC": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    probability_spread = (
        comparison["Churn probability"].max()
        - comparison["Churn probability"].min()
    )
    col_1, col_2, col_3 = st.columns(3)
    col_1.metric(
        "Model agreement",
        (
            "Đồng thuận"
            if comparison["Prediction"].nunique() == 1
            else "Có bất đồng"
        ),
    )
    col_2.metric("Probability spread", f"{probability_spread:.1%}")
    col_3.metric(
        "Highest-risk model",
        comparison.iloc[0]["Algorithm"],
    )

with eda_tab:
    st.markdown(
        '<div class="section-title">EDA explorer — source train data</div>',
        unsafe_allow_html=True,
    )
    render_eda(demo_bundle)

with model_tab:
    st.markdown(
        '<div class="section-title">Frozen model bundle</div>',
        unsafe_allow_html=True,
    )
    model_overview = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Algorithm": artifact.algorithm,
                "Trees": artifact.best_iteration,
                "Imbalance": artifact.imbalance_strategy,
                "CV PR-AUC": artifact.cv_pr_auc_mean,
                "Fit rows": artifact.training_rows,
                "Fit seconds": artifact.training_seconds,
            }
            for model_name, artifact in demo_bundle.artifacts.items()
        ]
    )
    st.dataframe(
        model_overview,
        width="stretch",
        hide_index=True,
        column_config={
            "CV PR-AUC": st.column_config.NumberColumn(format="%.4f"),
            "Fit rows": st.column_config.NumberColumn(format="%d"),
            "Fit seconds": st.column_config.NumberColumn(format="%.2f s"),
        },
    )

    chosen_artifact = demo_bundle.artifacts[selected_model_name]
    parameter_column, pipeline_column = st.columns([1, 1])
    with parameter_column:
        st.markdown(f"#### Parameters — {selected_model_name}")
        st.json(chosen_artifact.best_parameters)
    with pipeline_column:
        st.markdown("#### Inference flow")
        st.code(
            """Raw customer
  → ChurnFeatureEngineer
  → Median imputation / One-hot encoding
  → Selected tuned model
  → P(Exited = 1)
  → Threshold decision""",
            language="text",
        )
        st.info(
            "CustomerId, RowNumber, Surname, Complain và Exited "
            "không đi vào feature matrix."
        )

    with st.expander("Xem feature engineering catalog"):
        st.dataframe(
            demo_bundle.features.catalog,
            width="stretch",
            hide_index=True,
        )

    st.warning(
        "Đây là application phục vụ demo. Probability cần được theo dõi "
        "calibration và drift trước khi dùng cho quyết định production."
    )
