"""
dashboard/app_v5_3.py

AI 智慧心血管疾病風險預測與健康管理系統
六大模組展示版儀表板 v5.3

使用前請先執行：
    python 05_generate_dashboard_insights.py

啟動方式：
    python -m streamlit run dashboard/app_v5_3.py

定位：
本系統為課程專題 MVP 與健康風險評估 / 決策支援原型，
不取代醫師診斷、醫療處置或正式臨床判斷。
"""

from pathlib import Path
from datetime import datetime, date, time
import json
import sqlite3

import joblib
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None


# =========================================================
# 1. 路徑設定
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_heart_data.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_risk_model.pkl"
FEATURE_SCHEMA_PATH = BASE_DIR / "models" / "feature_schema.json"
DB_PATH = BASE_DIR / "database" / "heart_disease_project.db"

REPORTS_DIR = BASE_DIR / "reports"
INSIGHTS_DIR = REPORTS_DIR / "dashboard_insights"

MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
USER_PREDICTION_PATH = REPORTS_DIR / "user_prediction_result.csv"
HEALTH_LOG_PATH = REPORTS_DIR / "health_management_logs.csv"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


# =========================================================
# 2. 頁面與視覺設計
# =========================================================

st.set_page_config(
    page_title="AI 心血管健康管理系統 v5",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_MAP = {
    "blue": "#2563EB",
    "teal": "#0F766E",
    "purple": "#7C3AED",
    "orange": "#F97316",
    "pink": "#DB2777",
    "green": "#16A34A",
}

STATUS_COLOR_MAP = {
    "穩定觀察": "#16A34A",
    "建議加強追蹤": "#F97316",
    "建議優先關注": "#DC2626",
    "未分類": "#64748B",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif;
    }

    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .stApp {
        background:
            radial-gradient(circle at 6% 8%, rgba(14, 165, 233, 0.24), transparent 28%),
            radial-gradient(circle at 92% 12%, rgba(168, 85, 247, 0.20), transparent 30%),
            radial-gradient(circle at 78% 88%, rgba(244, 114, 182, 0.16), transparent 28%),
            radial-gradient(circle at 16% 86%, rgba(34, 197, 94, 0.15), transparent 25%),
            linear-gradient(135deg, #F8FBFF 0%, #EEF6FF 34%, #FFF7ED 68%, #FDF2F8 100%);
        background-attachment: fixed;
    }

    .main {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94));
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }



    [data-testid="stSidebar"] .stRadio label {
        padding: 7px 10px;
        border-radius: 12px;
        margin-bottom: 3px;
        transition: 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.10);
    }

    .module-chip-row {
        margin-top: 10px;
        margin-bottom: 14px;
    }

    .module-chip {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.20);
        border: 1px solid rgba(255,255,255,0.28);
        color: white;
        font-size: 13px;
        font-weight: 800;
        margin: 4px 5px 0 0;
        backdrop-filter: blur(8px);
    }

    .hero {
        padding: 34px 38px;
        border-radius: 32px;
        color: white;
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 42%, #7C3AED 72%, #DB2777 100%);
        box-shadow: 0 26px 60px rgba(37, 99, 235, 0.26);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 38px;
        line-height: 1.25;
        font-weight: 900;
        margin: 0 0 12px 0;
    }

    .hero p {
        font-size: 17px;
        line-height: 1.75;
        opacity: 0.96;
        margin: 0;
    }

    .section-title {
        font-size: 25px;
        font-weight: 900;
        color: #0F172A;
        margin: 18px 0 12px 0;
    }

    .sub-text {
        color: #64748B;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 12px;
    }

    .project-card {
        padding: 18px 18px;
        border-radius: 22px;
        color: white;
        min-height: 165px;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.12);
        margin-bottom: 14px;
    }

    .project-card h3 {
        font-size: 20px;
        font-weight: 900;
        margin: 0 0 9px 0;
    }

    .project-card p {
        font-size: 13px;
        line-height: 1.6;
        margin: 0;
        opacity: 0.95;
    }

    .soft-card {
        padding: 19px 21px;
        border-radius: 22px;
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(226,232,240,0.95);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
    }

    .status-stable {
        padding: 18px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #DCFCE7, #ECFDF5);
        border: 1px solid #86EFAC;
        color: #065F46;
        font-weight: 800;
        margin: 12px 0;
    }

    .status-follow {
        padding: 18px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #FEF3C7, #FFFBEB);
        border: 1px solid #FCD34D;
        color: #92400E;
        font-weight: 800;
        margin: 12px 0;
    }

    .status-priority {
        padding: 18px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #FEE2E2, #FEF2F2);
        border: 1px solid #FCA5A5;
        color: #991B1B;
        font-weight: 800;
        margin: 12px 0;
    }

    .mini-pill {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 5px 4px 0;
        border-radius: 999px;
        color: white;
        font-size: 13px;
        font-weight: 700;
    }

    .flow-box {
        padding: 14px 16px;
        border-radius: 16px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E3A8A;
        text-align: center;
        font-weight: 800;
        margin-bottom: 8px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E5E7EB;
        padding: 16px 18px;
        border-radius: 20px;
        box-shadow: 0 9px 26px rgba(15, 23, 42, 0.055);
    }



    .quick-action-card {
        padding: 18px 18px;
        border-radius: 22px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(226,232,240,0.95);
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
        margin-bottom: 14px;
    }

    .sync-card {
        padding: 17px 19px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(20,184,166,0.10));
        border: 1px solid rgba(59,130,246,0.25);
        color: #0F172A;
        line-height: 1.7;
        margin-bottom: 16px;
    }

    section[data-testid="stSidebar"] button {
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
        background: rgba(255,255,255,0.82) !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] button:hover {
        border-color: rgba(37, 99, 235, 0.75) !important;
        background: linear-gradient(135deg, rgba(37,99,235,0.13), rgba(124,58,237,0.12)) !important;
        transform: translateY(-1px);
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 13px;
        padding: 20px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. 載入資料與工具
# =========================================================

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_dynamic_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_schema() -> dict:
    if FEATURE_SCHEMA_PATH.exists():
        with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"feature_columns": FEATURE_COLUMNS}


def insight(name: str) -> pd.DataFrame:
    return load_csv(INSIGHTS_DIR / name)


def money(value) -> str:
    try:
        return f"NT$ {int(value):,}"
    except Exception:
        return str(value)


def combine_date_time(record_date: date, record_time: time) -> str:
    return datetime.combine(record_date, record_time).strftime("%Y-%m-%d %H:%M:%S")


def parse_datetime_column(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")
    if "record_datetime" in df.columns:
        return pd.to_datetime(df["record_datetime"], errors="coerce")
    if "created_at" in df.columns:
        return pd.to_datetime(df["created_at"], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def filter_date_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty:
        return df

    temp = df.copy()
    temp["_dt"] = parse_datetime_column(temp)

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    temp = temp[(temp["_dt"] >= start_dt) & (temp["_dt"] <= end_dt)]
    return temp.drop(columns=["_dt"])


def metric_value(kpi_df: pd.DataFrame, name: str, default="0"):
    if kpi_df.empty or "kpi" not in kpi_df.columns:
        return default

    row = kpi_df[kpi_df["kpi"] == name]
    if row.empty:
        return default

    return row.iloc[0]["value"]


def status_from_score(score: float) -> str:
    if score < 0.30:
        return "stable"
    if score < 0.70:
        return "follow"
    return "priority"


def public_status_label(status: str) -> str:
    mapping = {
        "stable": "目前狀態：穩定觀察",
        "follow": "目前狀態：建議加強追蹤",
        "priority": "目前狀態：建議優先諮詢醫療人員",
    }
    return mapping.get(status, "目前狀態：需進一步確認")


def internal_risk_label(status: str) -> str:
    mapping = {
        "stable": "低風險",
        "follow": "中風險",
        "priority": "高風險",
    }
    return mapping.get(status, "未分類")


def status_box(status: str, text: str) -> None:
    class_name = {
        "stable": "status-stable",
        "follow": "status-follow",
        "priority": "status-priority",
    }.get(status, "status-follow")

    st.markdown(f"<div class='{class_name}'>{text}</div>", unsafe_allow_html=True)


def bp_label(systolic: float, diastolic: float = 80) -> str:
    if systolic < 120 and diastolic < 80:
        return "血壓標記：目前紀錄較穩定"
    if systolic < 130:
        return "血壓標記：建議持續觀察"
    if systolic < 140:
        return "血壓標記：偏高，建議規律追蹤"
    return "血壓標記：偏高，建議加強追蹤並視情況諮詢醫療人員"


def build_suggestions(input_data: dict, status: str) -> list:
    suggestions = []

    if status == "stable":
        suggestions.append("建議維持規律作息、均衡飲食與定期健康追蹤。")
    elif status == "follow":
        suggestions.append("建議持續追蹤血壓、血脂與血糖，必要時安排健康檢查。")
    else:
        suggestions.append("建議優先諮詢醫療人員，進一步評估心血管相關風險。")

    if input_data.get("trestbps", 0) >= 140:
        suggestions.append("靜息血壓偏高，建議規律量測並記錄趨勢。")

    if input_data.get("chol", 0) >= 240:
        suggestions.append("膽固醇偏高，建議追蹤血脂並留意飲食調整。")

    if input_data.get("fbs", 0) == 1:
        suggestions.append("空腹血糖指標異常，建議追蹤血糖與代謝相關風險。")

    if input_data.get("restecg", 0) in [1, 2]:
        suggestions.append("靜息心電圖顯示需留意的標記，建議由醫療人員協助判讀。")

    if input_data.get("exang", 0) == 1:
        suggestions.append("有運動誘發心絞痛訊號，建議不要忽視症狀，必要時尋求醫療評估。")

    if input_data.get("oldpeak", 0) >= 2:
        suggestions.append("oldpeak 數值較高，建議搭配心電圖或醫療人員判讀。")

    suggestions.append("本結果僅作為健康風險評估與課程展示，不代表正式診斷。")
    return suggestions


def create_tables() -> None:
    if not DB_PATH.exists():
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS web_prediction_records_v5 (
            record_id TEXT,
            created_at TEXT,
            record_datetime TEXT,
            user_name TEXT,
            age REAL,
            sex REAL,
            cp REAL,
            trestbps REAL,
            chol REAL,
            fbs REAL,
            restecg REAL,
            thalach REAL,
            exang REAL,
            oldpeak REAL,
            slope REAL,
            ca REAL,
            thal REAL,
            risk_score REAL,
            internal_risk_level TEXT,
            public_status TEXT,
            suggestion_text TEXT
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS health_management_logs_v5 (
            log_id TEXT,
            created_at TEXT,
            record_datetime TEXT,
            user_name TEXT,
            systolic_bp REAL,
            diastolic_bp REAL,
            blood_sugar REAL,
            weight REAL,
            exercise_minutes REAL,
            sleep_hours REAL,
            symptom_note TEXT,
            bp_label TEXT
        )
        """)

        conn.commit()


def save_prediction(record_datetime: str, user_name: str, input_data: dict,
                    risk_score: float, status: str, suggestions: list) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    create_tables()

    record_id = "WEB_" + datetime.now().strftime("%Y%m%d%H%M%S")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "record_id": record_id,
        "created_at": created_at,
        "record_datetime": record_datetime,
        "user_name": user_name,
        **input_data,
        "risk_score": risk_score,
        "internal_risk_level": internal_risk_label(status),
        "public_status": public_status_label(status),
        "suggestion_text": "；".join(suggestions),
    }

    row_df = pd.DataFrame([row])

    if USER_PREDICTION_PATH.exists():
        old_df = pd.read_csv(USER_PREDICTION_PATH)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df

    new_df.to_csv(USER_PREDICTION_PATH, index=False, encoding="utf-8-sig")

    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            row_df.to_sql("web_prediction_records_v5", conn, if_exists="append", index=False)

    return record_id


def save_health_log(record_datetime: str, user_name: str, log_data: dict) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    create_tables()

    log_id = "HL_" + datetime.now().strftime("%Y%m%d%H%M%S")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = bp_label(log_data["systolic_bp"], log_data["diastolic_bp"])

    row = {
        "log_id": log_id,
        "created_at": created_at,
        "record_datetime": record_datetime,
        "user_name": user_name,
        **log_data,
        "bp_label": label,
    }

    row_df = pd.DataFrame([row])

    if HEALTH_LOG_PATH.exists():
        old_df = pd.read_csv(HEALTH_LOG_PATH)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df

    new_df.to_csv(HEALTH_LOG_PATH, index=False, encoding="utf-8-sig")

    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            row_df.to_sql("health_management_logs_v5", conn, if_exists="append", index=False)

    return log_id


def get_tables() -> list:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
    return df["name"].tolist()


def read_sqlite_table(table_name: str, limit: int = 100) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)


def simulate_future(model, feature_columns: list, input_data: dict) -> pd.DataFrame:
    current = input_data.copy()

    improved = input_data.copy()
    improved["trestbps"] = max(100, improved["trestbps"] - 10)
    improved["chol"] = max(130, improved["chol"] - 20)
    improved["oldpeak"] = max(0, improved["oldpeak"] - 0.5)
    improved["fbs"] = 0 if improved["fbs"] == 1 else improved["fbs"]

    worsened = input_data.copy()
    worsened["trestbps"] = min(260, worsened["trestbps"] + 10)
    worsened["chol"] = min(700, worsened["chol"] + 20)
    worsened["oldpeak"] = min(10, worsened["oldpeak"] + 0.5)
    worsened["exang"] = 1

    rows = [
        ("現在", "依目前輸入資料評估", current),
        ("30 天後", "若維持追蹤與生活型態管理", current),
        ("60 天後", "追蹤改善情境：血壓、膽固醇或部分指標改善", improved),
        ("90 天後", "未追蹤且部分指標惡化情境", worsened),
    ]

    result = []
    for period, scenario, data in rows:
        score = float(model.predict_proba(pd.DataFrame([data], columns=feature_columns))[:, 1][0])
        status = status_from_score(score)
        result.append({
            "時間": period,
            "情境": scenario,
            "risk_score": score,
            "目前狀態": public_status_label(status),
        })

    return pd.DataFrame(result)


def plot_bar(df, x, y, title, color=None, orientation="v"):
    if df.empty:
        st.warning("目前沒有資料可繪圖。")
        return

    if px is None:
        st.bar_chart(df.set_index(x)[y])
        return

    if orientation == "h":
        fig = px.bar(df, x=y, y=x, color=color, orientation="h", title=title)
    else:
        fig = px.bar(df, x=x, y=y, color=color, title=title)

    fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_line(df, x, y, title, color=None):
    if df.empty:
        st.warning("目前沒有資料可繪圖。")
        return

    if px is None:
        st.line_chart(df.set_index(x)[y])
        return

    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_pie(df, names, values, title):
    if df.empty:
        st.warning("目前沒有資料可繪圖。")
        return

    if px is None:
        st.dataframe(df, use_container_width=True)
        return

    fig = px.pie(df, names=names, values=values, hole=0.45, title=title)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 4. 載入資料
# =========================================================

heart_df = load_csv(DATA_PATH)
model_df = load_csv(MODEL_COMPARISON_PATH)
user_df = load_dynamic_csv(USER_PREDICTION_PATH)
health_df = load_dynamic_csv(HEALTH_LOG_PATH)

model = load_model()
schema = load_schema()
feature_columns = schema.get("feature_columns", FEATURE_COLUMNS)

subproject_df = insight("subproject_overview.csv")
kpi_df = insight("dashboard_kpi_summary.csv")
target_df = insight("target_distribution.csv")
quality_df = insight("data_quality_overview.csv")
group_risk_df = insight("clinical_group_risk_summary.csv")
clinical_cat_df = insight("clinical_categorical_risk_table.csv")
clinical_num_df = insight("clinical_numeric_compare_table.csv")
corr_df = insight("clinical_correlation_matrix.csv")
model_summary_df = insight("model_dashboard_summary.csv")
model_for_dashboard_df = insight("model_comparison_for_dashboard.csv")
feature_importance_df = insight("model_feature_importance.csv")
cost_df = insight("cost_effectiveness_demo.csv")
resource_df = insight("resource_allocation_demo.csv")
status_df = insight("user_status_distribution.csv")
health_alert_df = insight("health_alert_summary.csv")

if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None

if "health_df" not in st.session_state:
    st.session_state["health_df"] = health_df

if "health_prefill" not in st.session_state:
    st.session_state["health_prefill"] = None

NAV_PAGES = [
    "系統首頁",
    "使用者輸入｜AI 評估",
    "資料中心｜整理狀態",
    "臨床洞察｜指標分析",
    "AI 核心｜模型分析",
    "決策支援｜成本效益",
    "流程管理｜分流儀表板",
    "健康管理｜追蹤提醒",
    "歷史紀錄",
    "資料庫導覽",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "系統首頁"

if st.session_state["nav_page"] not in NAV_PAGES:
    st.session_state["nav_page"] = "系統首頁"


def go_to_page(page_name: str) -> None:
    st.session_state["nav_page"] = page_name
    st.rerun()


# =========================================================
# 5. 側邊欄
# =========================================================

st.sidebar.markdown("### ❤️ HeartHealth Hub")
st.sidebar.caption("AI 風險評估 × 健康追蹤 × 管理分析")
st.sidebar.markdown("---")
st.sidebar.caption("功能導覽")

for item in NAV_PAGES:
    icon = "●" if st.session_state["nav_page"] == item else "○"
    if st.sidebar.button(f"{icon} {item}", key=f"nav_{item}", use_container_width=True):
        st.session_state["nav_page"] = item
        st.rerun()

page = st.session_state["nav_page"]

st.sidebar.markdown("---")
st.sidebar.caption("目前 MVP 狀態")
st.sidebar.write(f"訓練資料：{len(heart_df) if not heart_df.empty else 0} 筆")
st.sidebar.write(f"AI 評估：{len(user_df) if not user_df.empty else 0} 筆")
st.sidebar.write(f"健康追蹤：{len(st.session_state['health_df']) if not st.session_state['health_df'].empty else 0} 筆")


# =========================================================
# 6. 系統首頁
# =========================================================

if page == "系統首頁":
    st.markdown(
        """
        <div class="hero">
            <h1>AI 智慧心血管疾病風險預測與健康管理系統</h1>
            <p>
            六大功能模組整合展示版：從資料清理、臨床分析、AI 模型、成本效益、使用者分流，
            到健康追蹤與管理者儀表板。本系統為課程專題與健康風險評估原型，不取代醫師診斷。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleveland 訓練資料", len(heart_df) if not heart_df.empty else 0)
    c2.metric("比較模型數", len(model_for_dashboard_df) if not model_for_dashboard_df.empty else len(model_df))
    c3.metric("AI 評估紀錄", int(metric_value(kpi_df, "prediction_count", 0)))
    c4.metric("健康追蹤紀錄", int(metric_value(kpi_df, "health_log_count", 0)))

    st.markdown('<div class="section-title">快速入口</div>', unsafe_allow_html=True)
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        st.markdown('<div class="quick-action-card"><b>🤖 AI 評估</b><br><span class="sub-text">輸入資料並產生目前狀態與建議。</span></div>', unsafe_allow_html=True)
        if st.button("開始 AI 評估", key="home_go_ai", use_container_width=True):
            go_to_page("使用者輸入｜AI 評估")
    with qa2:
        st.markdown('<div class="quick-action-card"><b>🫀 健康追蹤</b><br><span class="sub-text">接續最新 AI 評估，紀錄血壓、血糖、體重與生活型態。</span></div>', unsafe_allow_html=True)
        if st.button("新增健康追蹤", key="home_go_health", use_container_width=True):
            go_to_page("健康管理｜追蹤提醒")
    with qa3:
        st.markdown('<div class="quick-action-card"><b>📊 模型分析</b><br><span class="sub-text">查看 303 筆資料與三種模型比較。</span></div>', unsafe_allow_html=True)
        if st.button("查看 AI 核心", key="home_go_model", use_container_width=True):
            go_to_page("AI 核心｜模型分析")
    with qa4:
        st.markdown('<div class="quick-action-card"><b>💡 決策支援</b><br><span class="sub-text">查看檢查組合、模擬價格與資源配置。</span></div>', unsafe_allow_html=True)
        if st.button("查看成本效益", key="home_go_cost", use_container_width=True):
            go_to_page("決策支援｜成本效益")

    st.markdown('<div class="section-title">六大功能模組進度地圖</div>', unsafe_allow_html=True)

    if subproject_df.empty:
        st.warning("找不到 subproject_overview.csv。請先執行 python 05_generate_dashboard_insights.py")
    else:
        cols = st.columns(3)

        for index, row in subproject_df.iterrows():
            color = COLOR_MAP.get(row.get("color_group", "blue"), "#2563EB")
            with cols[index % 3]:
                st.markdown(
                    f"""
                    <div class="project-card" style="background: linear-gradient(135deg, {color}, #111827);">
                        <h3>模組 {int(row['subproject_id'])}｜{row['short_name']}</h3>
                        <p><b>{row['status']}</b></p>
                        <p>{row['subproject_name']}</p>
                        <p style="margin-top:8px;">輸出：{row['main_outputs']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-title">系統資料流</div>', unsafe_allow_html=True)

    flow_cols = st.columns(6)
    flow_items = ["Excel 資料", "CSV / SQLite", "臨床分析", "AI 模型", "使用者評估", "健康管理"]
    for col, item in zip(flow_cols, flow_items):
        with col:
            st.markdown(f"<div class='flow-box'>{item}</div>", unsafe_allow_html=True)

    st.warning("提醒：risk_score 僅作健康風險評估與決策支援展示，不代表正式診斷。")


# =========================================================
# 7. 使用者輸入｜AI 評估
# =========================================================

elif page == "使用者輸入｜AI 評估":
    st.title("使用者輸入｜AI 即時評估")
    st.markdown(
        """
        <div class="soft-card">
        這一頁是給使用者操作的入口：輸入姓名、日期時間與檢查資料後，系統會立即產生 risk_score、目前狀態與下一步建議。<br>
        這裡保留原本你想要的「使用者可以直接輸入並得到結果」流程，不再藏在模型分析頁底下。
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 即時 AI 評估")
    if model is None:
        st.error("找不到模型檔案。")
    else:
        with st.form("prediction_form_v5_1_user"):
            t1, t2, t3 = st.columns([2, 1, 1])
            with t1:
                user_name = st.text_input("姓名 / 暱稱", value="測試使用者")
            with t2:
                record_date = st.date_input("評估日期", value=date.today())
            with t3:
                record_time = st.time_input("評估時間", value=datetime.now().time().replace(second=0, microsecond=0))

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("年齡", min_value=1, max_value=120, value=63)
                sex_text = st.selectbox("性別", ["男性", "女性"])
                cp = st.selectbox("胸痛型態", [1, 2, 3, 4], format_func=lambda x: {1:"典型心絞痛", 2:"非典型心絞痛", 3:"非心絞痛疼痛", 4:"無症狀"}[x])
                trestbps = st.number_input("靜息血壓", min_value=1, max_value=260, value=145)

            with col2:
                chol = st.number_input("膽固醇", min_value=1, max_value=700, value=233)
                fbs = st.selectbox("空腹血糖 > 120", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
                restecg = st.selectbox("靜息心電圖", [0, 1, 2], format_func=lambda x: {0:"正常", 1:"ST-T異常", 2:"左心室肥大可能"}[x])
                thalach = st.number_input("最大心率", min_value=1, max_value=250, value=150)

            with col3:
                exang = st.selectbox("運動誘發心絞痛", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
                oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
                slope = st.selectbox("ST 斜率", [1, 2, 3], format_func=lambda x: {1:"上升", 2:"平坦", 3:"下降"}[x])
                ca = st.selectbox("螢光顯影血管數 ca", [0, 1, 2, 3])
                thal = st.selectbox("Thal", [3, 6, 7], format_func=lambda x: {3:"正常", 6:"固定缺陷", 7:"可逆缺陷"}[x])

            submitted = st.form_submit_button("開始 AI 評估")

        if submitted:
            input_data = {
                "age": float(age),
                "sex": 1 if sex_text == "男性" else 0,
                "cp": int(cp),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": int(fbs),
                "restecg": int(restecg),
                "thalach": float(thalach),
                "exang": int(exang),
                "oldpeak": float(oldpeak),
                "slope": int(slope),
                "ca": int(ca),
                "thal": int(thal),
            }

            record_datetime = combine_date_time(record_date, record_time)
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            risk_score = float(model.predict_proba(input_df)[:, 1][0])
            status = status_from_score(risk_score)
            suggestions = build_suggestions(input_data, status)
            record_id = save_prediction(record_datetime, user_name, input_data, risk_score, status, suggestions)

            st.session_state["latest_prediction"] = {
                "record_id": record_id,
                "record_datetime": record_datetime,
                "user_name": user_name,
                "input_data": input_data,
                "risk_score": risk_score,
                "status": status,
                "suggestions": suggestions,
            }
            st.session_state["health_prefill"] = {
                "source_record_id": record_id,
                "record_datetime": record_datetime,
                "user_name": user_name,
                "systolic_bp": float(input_data.get("trestbps", 120)),
                "diastolic_bp": 80.0,
                "blood_sugar": 126.0 if int(input_data.get("fbs", 0)) == 1 else 0.0,
                "risk_score": risk_score,
                "status": status,
            }

            r1, r2, r3 = st.columns(3)
            r1.metric("紀錄編號", record_id)
            r2.metric("risk_score", f"{risk_score:.4f}")
            r3.metric("目前狀態", public_status_label(status).replace("目前狀態：", ""))

            status_box(status, public_status_label(status))

            st.markdown("#### 下一步建議")
            for index, suggestion in enumerate(suggestions, start=1):
                st.write(f"{index}. {suggestion}")

            st.markdown("---")
            st.markdown("#### 接續健康管理")
            st.write("這筆 AI 評估已暫存為健康管理預填資料，可直接帶入姓名、日期、靜息血壓與血糖標記。")
            if st.button("同步到健康管理追蹤", key=f"sync_health_{record_id}", use_container_width=True):
                go_to_page("健康管理｜追蹤提醒")



# =========================================================
# 7. 資料中心｜整理狀態
# =========================================================

elif page == "資料中心｜整理狀態":
    st.title("資料中心｜資料整理與資料庫狀態")

    st.markdown(
        """
        <div class="sub-text">
        這一頁用來確認資料有沒有整理成功。包含：總共有幾筆資料、模型訓練用的 0/1 分類比例、哪些欄位有缺資料，以及 SQLite 資料庫目前有哪些表格。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("說明：0/1 分類是模型訓練用的目標欄位。0 代表資料中沒有心臟病紀錄，1 代表資料中有心臟病風險紀錄；這只是資料集標記，不是本系統對使用者做正式診斷。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總資料筆數", len(heart_df) if not heart_df.empty else 0)
    c2.metric("資料欄位數", heart_df.shape[1] if not heart_df.empty else 0)

    if not target_df.empty and "count" in target_df.columns:
        c3.metric("0 類紀錄", int(target_df[target_df["target"] == 0]["count"].sum()))
        c4.metric("1 類風險紀錄", int(target_df[target_df["target"] == 1]["count"].sum()))
    else:
        c3.metric("0 類紀錄", "-")
        c4.metric("1 類風險紀錄", "-")

    left, right = st.columns(2)

    with left:
        plot_pie(target_df, "target_label", "count", "模型訓練分類比例（0/1）")

    with right:
        if not quality_df.empty:
            missing_show = quality_df.sort_values("missing_count", ascending=False).head(12)
            plot_bar(missing_show, "column_name", "missing_count", "缺漏資料統計", orientation="h")
        else:
            st.warning("找不到資料品質摘要。")

    st.markdown("### 不同族群的風險紀錄比例")
    if not group_risk_df.empty:
        group_display_df = group_risk_df.rename(columns={
            "dimension": "分組類型",
            "group": "分組",
            "count": "分組總筆數",
            "risk_count": "風險紀錄數（1 類風險紀錄）",
            "risk_rate": "風險紀錄比例（%）",
        })
        plot_bar(group_display_df, "分組", "風險紀錄比例（%）", "年齡、血壓、膽固醇分組的風險紀錄比例", color="分組類型")
        st.dataframe(group_display_df, use_container_width=True)
    else:
        st.info("目前資料中沒有 age_group、bp_group 或 chol_group，或尚未產生摘要。")

    st.markdown("### 資料庫表格")
    tables = get_tables()
    st.write(tables if tables else "目前讀不到 資料庫表格。")


# =========================================================
# 8. 臨床洞察｜指標分析
# =========================================================

elif page == "臨床洞察｜指標分析":
    st.title("臨床洞察｜哪些指標和風險比較有關")

    st.markdown(
        """
        <div class="sub-text">
        這一頁用來看「哪些檢查或身體指標，和心血管風險紀錄比較有關」。例如胸痛型態、血壓、膽固醇、心電圖、運動誘發心絞痛等，會被整理成圖表，作為健康建議的依據。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("看圖小提醒：風險紀錄比例越高，代表該分組在 Cleveland 資料中 target=1 的比例越高；這可以作為健康建議依據，但不能單獨解讀成因果關係。")

    if clinical_cat_df.empty:
        st.warning("找不到 clinical_categorical_risk_table.csv。")
    else:
        selected_feature = st.selectbox(
            "選擇要看的指標",
            clinical_cat_df["feature"].dropna().unique().tolist(),
            index=0,
        )

        show_df = clinical_cat_df[clinical_cat_df["feature"] == selected_feature].copy()
        show_display_df = show_df.rename(columns={
            "feature": "欄位",
            "value": "原始值",
            "label": "類別標記",
            "sample_count": "資料筆數",
            "risk_count": "風險紀錄數（1 類風險紀錄）",
            "risk_rate": "風險紀錄比例（%）",
        })

        left, right = st.columns([1.15, 1])

        with left:
            plot_bar(show_display_df, "類別標記", "風險紀錄比例（%）", f"{selected_feature} 各類別風險紀錄比例", color="類別標記")

        with right:
            plot_bar(show_display_df, "類別標記", "資料筆數", f"{selected_feature} 資料筆數", color="類別標記")

        st.dataframe(show_display_df, use_container_width=True)

    st.markdown("### 數值指標比較：有風險紀錄 vs 無風險紀錄")
    if not clinical_num_df.empty:
        show_num_df = clinical_num_df.sort_values("difference", ascending=False)
        plot_bar(show_num_df, "feature", "difference", "兩組平均值差異", color="feature")
        show_num_display_df = show_num_df.rename(columns={
            "feature": "欄位",
            "no_risk_mean": "無風險紀錄平均值（0 類紀錄）",
            "risk_mean": "風險紀錄平均值（1 類風險紀錄）",
            "difference": "平均差異",
            "no_risk_median": "無風險紀錄中位數",
            "risk_median": "風險紀錄中位數",
        })
        st.dataframe(show_num_display_df, use_container_width=True)

    st.markdown("### 指標關聯熱力圖")
    if corr_df.empty:
        st.warning("找不到 correlation matrix。")
    else:
        heat_df = corr_df.set_index("feature")

        if px is None:
            st.dataframe(heat_df.style.background_gradient(cmap="RdBu", axis=None).format("{:.2f}"), use_container_width=True)
        else:
            fig = px.imshow(
                heat_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="健康指標關聯熱力圖",
                aspect="auto",
            )
            fig.update_layout(height=650, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 9. AI 核心｜模型分析
# =========================================================

elif page == "AI 核心｜模型分析":
    st.title("AI 核心｜模型表現與預測依據")

    st.markdown(
        """
        <div class="soft-card">
        <b>重要說明：</b>303 筆是 Cleveland 清理後資料的總筆數；3 個是比較的模型演算法數量，不是只拿 3 筆資料訓練。這一頁會說明資料如何用來訓練模型、哪個模型表現最好，以及模型比較重視哪些指標。
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_summary_df.empty:
        summary_dict = dict(zip(model_summary_df["metric"], model_summary_df["value"]))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總總資料筆數", summary_dict.get("dataset_total_rows", "-"))
        c2.metric("訓練筆數約", summary_dict.get("train_rows_estimated", "-"))
        c3.metric("測試筆數約", summary_dict.get("test_rows_estimated", "-"))
        c4.metric("最佳模型", summary_dict.get("best_model", "-"))

        c5, c6 = st.columns(2)
        c5.metric("比較模型數", summary_dict.get("model_count", "-"))
        c6.metric("最佳 AUC", summary_dict.get("best_auc", "-"))

    left, right = st.columns(2)

    with left:
        if not model_for_dashboard_df.empty and "auc" in model_for_dashboard_df.columns:
            plot_bar(model_for_dashboard_df, "model_name", "auc", "三種模型辨識能力比較（AUC）", color="model_name")
        else:
            st.warning("找不到模型比較資料。")

    with right:
        if not model_for_dashboard_df.empty:
            metric_cols = [col for col in ["accuracy", "precision", "recall", "f1", "auc"] if col in model_for_dashboard_df.columns]
            if metric_cols and px is not None:
                long_df = model_for_dashboard_df.melt(
                    id_vars="model_name",
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="score",
                )
                fig = px.bar(long_df, x="metric", y="score", color="model_name", barmode="group", title="模型表現比較總覽")
                fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(model_for_dashboard_df, use_container_width=True)

    st.markdown("### 特徵重要度")
    if not feature_importance_df.empty:
        show_imp = feature_importance_df.sort_values("importance", ascending=True).tail(12)
        plot_bar(show_imp, "feature", "importance", "模型最重視的指標", color="feature", orientation="h")
        st.dataframe(feature_importance_df, use_container_width=True)
    else:
        st.warning("找不到 model_feature_importance.csv。")

    st.markdown("### 即時 AI 評估")
    if model is None:
        st.error("找不到模型檔案。")
    else:
        with st.form("prediction_form_v5"):
            t1, t2, t3 = st.columns([2, 1, 1])
            with t1:
                user_name = st.text_input("姓名 / 暱稱", value="測試使用者")
            with t2:
                record_date = st.date_input("評估日期", value=date.today())
            with t3:
                record_time = st.time_input("評估時間", value=datetime.now().time().replace(second=0, microsecond=0))

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("年齡", min_value=1, max_value=120, value=63)
                sex_text = st.selectbox("性別", ["男性", "女性"])
                cp = st.selectbox("胸痛型態", [1, 2, 3, 4], format_func=lambda x: {1:"典型心絞痛", 2:"非典型心絞痛", 3:"非心絞痛疼痛", 4:"無症狀"}[x])
                trestbps = st.number_input("靜息血壓", min_value=1, max_value=260, value=145)

            with col2:
                chol = st.number_input("膽固醇", min_value=1, max_value=700, value=233)
                fbs = st.selectbox("空腹血糖 > 120", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
                restecg = st.selectbox("靜息心電圖", [0, 1, 2], format_func=lambda x: {0:"正常", 1:"ST-T異常", 2:"左心室肥大可能"}[x])
                thalach = st.number_input("最大心率", min_value=1, max_value=250, value=150)

            with col3:
                exang = st.selectbox("運動誘發心絞痛", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
                oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
                slope = st.selectbox("ST 斜率", [1, 2, 3], format_func=lambda x: {1:"上升", 2:"平坦", 3:"下降"}[x])
                ca = st.selectbox("螢光顯影血管數 ca", [0, 1, 2, 3])
                thal = st.selectbox("Thal", [3, 6, 7], format_func=lambda x: {3:"正常", 6:"固定缺陷", 7:"可逆缺陷"}[x])

            submitted = st.form_submit_button("開始 AI 評估")

        if submitted:
            input_data = {
                "age": float(age),
                "sex": 1 if sex_text == "男性" else 0,
                "cp": int(cp),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": int(fbs),
                "restecg": int(restecg),
                "thalach": float(thalach),
                "exang": int(exang),
                "oldpeak": float(oldpeak),
                "slope": int(slope),
                "ca": int(ca),
                "thal": int(thal),
            }

            record_datetime = combine_date_time(record_date, record_time)
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            risk_score = float(model.predict_proba(input_df)[:, 1][0])
            status = status_from_score(risk_score)
            suggestions = build_suggestions(input_data, status)
            record_id = save_prediction(record_datetime, user_name, input_data, risk_score, status, suggestions)

            st.session_state["latest_prediction"] = {
                "record_id": record_id,
                "record_datetime": record_datetime,
                "user_name": user_name,
                "input_data": input_data,
                "risk_score": risk_score,
                "status": status,
                "suggestions": suggestions,
            }
            st.session_state["health_prefill"] = {
                "source_record_id": record_id,
                "record_datetime": record_datetime,
                "user_name": user_name,
                "systolic_bp": float(input_data.get("trestbps", 120)),
                "diastolic_bp": 80.0,
                "blood_sugar": 126.0 if int(input_data.get("fbs", 0)) == 1 else 0.0,
                "risk_score": risk_score,
                "status": status,
            }

            r1, r2, r3 = st.columns(3)
            r1.metric("紀錄編號", record_id)
            r2.metric("risk_score", f"{risk_score:.4f}")
            r3.metric("目前狀態", public_status_label(status).replace("目前狀態：", ""))

            status_box(status, public_status_label(status))

            st.markdown("#### 下一步建議")
            for index, suggestion in enumerate(suggestions, start=1):
                st.write(f"{index}. {suggestion}")

            st.markdown("---")
            st.markdown("#### 接續健康管理")
            st.write("這筆 AI 評估已暫存為健康管理預填資料，可直接帶入姓名、日期、靜息血壓與血糖標記。")
            if st.button("同步到健康管理追蹤", key=f"sync_health_{record_id}", use_container_width=True):
                go_to_page("健康管理｜追蹤提醒")


# =========================================================
# 10. 決策支援｜成本效益
# =========================================================

elif page == "決策支援｜成本效益":
    st.title("決策支援｜檢查組合與資源配置")

    st.info("這一頁是「決策支援」展示。價格是課程 MVP 的模擬設定，不是醫院正式收費；主要用來示範不同檢查組合在成本、效益與資源配置上的取捨。")
    st.markdown(
        """
        <div class="soft-card">
        <b>建議組合價格怎麼來？</b><br>
        目前價格是 MVP 展示用的「模擬定價」，不是醫院或健檢中心正式收費。定價邏輯是依照檢查複雜度與資源消耗做階層設計：<br>
        低成本健康追蹤 → 基礎檢查 → 心電圖追蹤 → 完整心血管評估。<br>
        若要改價格，可以調整 <code>05_generate_dashboard_insights.py</code> 裡的 <code>generate_cost_effectiveness_demo()</code>，或直接修改 <code>reports/dashboard_insights/cost_effectiveness_demo.csv</code>。
        </div>
        """,
        unsafe_allow_html=True,
    )

    if cost_df.empty:
        st.warning("找不到 cost_effectiveness_demo.csv。")
    else:
        c1, c2 = st.columns(2)

        with c1:
            plot_bar(cost_df, "package_name", "estimated_cost", "模擬檢查組合價格比較", color="recommended_status")

        with c2:
            if px is None:
                st.dataframe(cost_df, use_container_width=True)
            else:
                fig = px.scatter(
                    cost_df,
                    x="estimated_cost",
                    y="expected_benefit",
                    size="benefit_per_1000_cost",
                    color="recommended_status",
                    hover_name="package_name",
                    title="模擬價格與效益比較",
                )
                fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 檢查組合建議")
        show_cost = cost_df.copy()
        show_cost["estimated_cost"] = show_cost["estimated_cost"].apply(money)
        st.dataframe(show_cost, use_container_width=True)

    st.markdown("### 資源配置模擬")
    if resource_df.empty:
        st.warning("找不到 resource_allocation_demo.csv。")
    else:
        plot_bar(resource_df, "status", "daily_slots", "每日檢查名額配置模擬", color="status")
        st.dataframe(resource_df, use_container_width=True)


# =========================================================
# 11. 流程管理｜分流儀表板
# =========================================================

elif page == "流程管理｜分流儀表板":
    st.title("流程管理｜從輸入到建議的系統流程")

    st.markdown("### 從使用者輸入到健康建議的流程")

    flow_cols = st.columns(6)
    flow_items = ["前台表單", "欄位驗證", "AI 預測", "狀態分流", "健康建議", "寫入儀表板"]
    for col, item in zip(flow_cols, flow_items):
        with col:
            st.markdown(f"<div class='flow-box'>{item}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("完成 AI 評估", int(metric_value(kpi_df, "prediction_count", 0)))
    c2.metric("平均 risk_score", metric_value(kpi_df, "average_risk_score", "-"))
    c3.metric("需優先關注", int(metric_value(kpi_df, "priority_attention_count", 0)))
    c4.metric("健康追蹤紀錄", int(metric_value(kpi_df, "health_log_count", 0)))

    left, right = st.columns(2)

    with left:
        plot_pie(status_df, "status_group", "count", "使用者目前狀態比例")

    with right:
        if not user_df.empty and "risk_score" in user_df.columns:
            show_user_df = user_df.copy()
            show_user_df["_dt"] = parse_datetime_column(show_user_df)
            show_user_df = show_user_df.dropna(subset=["_dt"]).sort_values("_dt")
            if not show_user_df.empty:
                plot_line(show_user_df, "_dt", "risk_score", "AI 評估分數歷史趨勢")
            else:
                st.warning("目前沒有可用時間欄位畫趨勢。")
        else:
            st.warning("目前尚無使用者預測紀錄。")

    st.markdown("### 最近 AI 評估紀錄")
    if user_df.empty:
        st.warning("目前尚無紀錄。")
    else:
        cols = [col for col in ["record_datetime", "created_at", "user_name", "risk_score", "public_status", "internal_risk_level", "suggestion_text"] if col in user_df.columns]
        st.dataframe(user_df.tail(30)[cols], use_container_width=True)


# =========================================================
# 12. 健康管理｜追蹤提醒
# =========================================================

elif page == "健康管理｜追蹤提醒":
    st.title("健康管理｜日常追蹤與提醒")

    st.markdown("### 新增一筆日常健康紀錄")

    latest = st.session_state.get("latest_prediction")
    prefill = st.session_state.get("health_prefill")

    default_name = "測試使用者"
    default_log_date = date.today()
    default_log_time = datetime.now().time().replace(second=0, microsecond=0)
    default_systolic = 120
    default_diastolic = 80
    default_blood_sugar = 0

    if latest is not None:
        default_name = latest.get("user_name", default_name)

    if prefill is not None:
        default_name = prefill.get("user_name", default_name)
        default_systolic = int(prefill.get("systolic_bp", default_systolic))
        default_diastolic = int(prefill.get("diastolic_bp", default_diastolic))
        default_blood_sugar = int(prefill.get("blood_sugar", default_blood_sugar))
        try:
            prefill_dt = pd.to_datetime(prefill.get("record_datetime"), errors="coerce")
            if pd.notna(prefill_dt):
                default_log_date = prefill_dt.date()
                default_log_time = prefill_dt.time().replace(second=0, microsecond=0)
        except Exception:
            pass

    if latest is not None:
        st.markdown(
            f"""
            <div class="sync-card">
            <b>已偵測到最新 AI 評估：</b>{latest.get('user_name', '')} ｜ risk_score：{float(latest.get('risk_score', 0)):.4f}<br>
            這一頁會自動帶入姓名與部分可追蹤欄位，例如靜息血壓與空腹血糖標記。你仍可手動修改成當日實際量測值。
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("health_form_v5"):
        t1, t2, t3 = st.columns([2, 1, 1])
        with t1:
            user_name = st.text_input("姓名 / 暱稱", value=default_name)
        with t2:
            log_date = st.date_input("紀錄日期", value=default_log_date)
        with t3:
            log_time = st.time_input("紀錄時間", value=default_log_time)

        c1, c2, c3 = st.columns(3)

        with c1:
            systolic_bp = st.number_input("收縮壓", min_value=50, max_value=260, value=default_systolic)
            diastolic_bp = st.number_input("舒張壓", min_value=30, max_value=160, value=default_diastolic)

        with c2:
            blood_sugar = st.number_input("血糖，未知可填 0", min_value=0, max_value=500, value=default_blood_sugar)
            weight = st.number_input("體重 kg", min_value=1.0, max_value=300.0, value=70.0, step=0.1)

        with c3:
            exercise_minutes = st.number_input("今日運動分鐘數", min_value=0, max_value=300, value=30)
            sleep_hours = st.number_input("睡眠時數", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

        symptom_note = st.text_area("症狀或備註，例如胸悶、胸痛、喘、心悸，可留空")

        submitted = st.form_submit_button("儲存健康追蹤")

    if submitted:
        record_datetime = combine_date_time(log_date, log_time)
        log_data = {
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "blood_sugar": float(blood_sugar),
            "weight": float(weight),
            "exercise_minutes": float(exercise_minutes),
            "sleep_hours": float(sleep_hours),
            "symptom_note": symptom_note,
        }

        log_id = save_health_log(record_datetime, user_name, log_data)
        st.session_state["health_df"] = load_dynamic_csv(HEALTH_LOG_PATH)

        st.success(f"健康追蹤已儲存：{log_id}")
        st.info(bp_label(float(systolic_bp), float(diastolic_bp)))

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    current_health_df = st.session_state["health_df"]
    c1.metric("追蹤總筆數", len(current_health_df) if not current_health_df.empty else 0)

    if not health_alert_df.empty:
        for i, row in health_alert_df.head(3).iterrows():
            if i == 0:
                c2.metric(row["alert_name"], int(row["count"]))
            elif i == 1:
                c3.metric(row["alert_name"], int(row["count"]))
            elif i == 2:
                c4.metric(row["alert_name"], int(row["count"]))

    left, right = st.columns(2)

    with left:
        st.markdown("### 異常提醒摘要")
        if health_alert_df.empty:
            st.warning("尚無提醒摘要。")
        else:
            plot_bar(health_alert_df, "alert_name", "count", "健康追蹤提醒摘要", color="alert_name")
            st.dataframe(health_alert_df, use_container_width=True)

    with right:
        st.markdown("### 健康趨勢圖")
        if current_health_df.empty:
            st.warning("目前尚無健康追蹤資料。")
        else:
            trend_cols = [col for col in ["systolic_bp", "diastolic_bp", "blood_sugar", "weight", "exercise_minutes", "sleep_hours"] if col in current_health_df.columns]
            show_health = current_health_df.copy()
            show_health["_dt"] = parse_datetime_column(show_health)
            show_health = show_health.dropna(subset=["_dt"]).sort_values("_dt")
            if trend_cols and not show_health.empty:
                if px is None:
                    st.line_chart(show_health[trend_cols].reset_index(drop=True))
                else:
                    long_df = show_health.melt(id_vars="_dt", value_vars=trend_cols, var_name="指標", value_name="數值")
                    fig = px.line(long_df, x="_dt", y="數值", color="指標", markers=True, title="健康追蹤趨勢")
                    fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 最近健康追蹤")
    if current_health_df.empty:
        st.warning("目前尚無健康追蹤紀錄。")
    else:
        st.dataframe(current_health_df.tail(30), use_container_width=True)


# =========================================================
# 13. 歷史紀錄
# =========================================================

elif page == "歷史紀錄":
    st.title("歷史紀錄查詢")

    tab1, tab2 = st.tabs(["AI 評估歷史", "健康追蹤歷史"])

    with tab1:
        history_df = load_dynamic_csv(USER_PREDICTION_PATH)

        if history_df.empty:
            st.warning("目前尚無 AI 評估歷史。")
        else:
            h1, h2, h3 = st.columns([2, 1, 1])
            with h1:
                keyword = st.text_input("搜尋姓名 / 暱稱", value="")
            with h2:
                start_date = st.date_input("開始日期", value=date.today().replace(day=1), key="pred_history_start")
            with h3:
                end_date = st.date_input("結束日期", value=date.today(), key="pred_history_end")

            show_df = history_df.copy()

            if keyword.strip() and "user_name" in show_df.columns:
                show_df = show_df[show_df["user_name"].astype(str).str.contains(keyword.strip(), case=False, na=False)]

            show_df = filter_date_range(show_df, start_date, end_date)

            if show_df.empty:
                st.warning("查無符合條件的 AI 評估紀錄。")
            else:
                st.metric("符合條件筆數", len(show_df))

                if "risk_score" in show_df.columns:
                    trend_df = show_df.copy()
                    trend_df["_dt"] = parse_datetime_column(trend_df)
                    trend_df = trend_df.dropna(subset=["_dt"]).sort_values("_dt")

                    if not trend_df.empty:
                        plot_line(trend_df, "_dt", "risk_score", "AI 評估歷史 risk_score 趨勢")

                cols = [col for col in ["record_datetime", "created_at", "user_name", "risk_score", "public_status", "internal_risk_level", "suggestion_text"] if col in show_df.columns]
                st.dataframe(show_df.tail(80)[cols], use_container_width=True)

    with tab2:
        health_history = load_dynamic_csv(HEALTH_LOG_PATH)

        if health_history.empty:
            st.warning("目前尚無健康追蹤歷史。")
        else:
            h1, h2, h3 = st.columns([2, 1, 1])
            with h1:
                names = health_history["user_name"].dropna().unique().tolist() if "user_name" in health_history.columns else []
                selected_name = st.selectbox("選擇使用者", names) if names else ""
            with h2:
                start_date = st.date_input("開始日期", value=date.today().replace(day=1), key="health_history_start")
            with h3:
                end_date = st.date_input("結束日期", value=date.today(), key="health_history_end")

            show_df = health_history.copy()

            if selected_name and "user_name" in show_df.columns:
                show_df = show_df[show_df["user_name"] == selected_name]

            show_df = filter_date_range(show_df, start_date, end_date)

            if show_df.empty:
                st.warning("查無符合條件的健康紀錄。")
            else:
                st.metric("符合條件筆數", len(show_df))

                trend_cols = [col for col in ["systolic_bp", "diastolic_bp", "blood_sugar", "weight", "exercise_minutes", "sleep_hours"] if col in show_df.columns]
                if trend_cols:
                    temp = show_df.copy()
                    temp["_dt"] = parse_datetime_column(temp)
                    temp = temp.dropna(subset=["_dt"]).sort_values("_dt")
                    if px is not None and not temp.empty:
                        long_df = temp.melt(id_vars="_dt", value_vars=trend_cols, var_name="指標", value_name="數值")
                        fig = px.line(long_df, x="_dt", y="數值", color="指標", markers=True, title="健康追蹤歷史趨勢")
                        fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    elif not temp.empty:
                        st.line_chart(temp[trend_cols].reset_index(drop=True))

                st.dataframe(show_df.tail(80), use_container_width=True)


# =========================================================
# 14. 資料庫導覽
# =========================================================

elif page == "資料庫導覽":
    st.title("資料庫導覽")

    tab1, tab2, tab3, tab4 = st.tabs(["資料產物", "欄位字典", "資料預覽", "SQLite"])

    with tab1:
        if subproject_df.empty:
            st.warning("找不到子專案總覽資料。")
        else:
            st.dataframe(subproject_df, use_container_width=True)

    with tab2:
        field_df = pd.DataFrame([
            {"欄位": "age", "說明": "年齡"},
            {"欄位": "sex", "說明": "性別，1=男性，0=女性"},
            {"欄位": "cp", "說明": "胸痛型態，1=典型心絞痛，2=非典型心絞痛，3=非心絞痛疼痛，4=無症狀"},
            {"欄位": "trestbps", "說明": "靜息血壓"},
            {"欄位": "chol", "說明": "膽固醇"},
            {"欄位": "fbs", "說明": "空腹血糖是否大於 120 mg/dl"},
            {"欄位": "restecg", "說明": "靜息心電圖，0=正常，1=ST-T異常，2=左心室肥大可能"},
            {"欄位": "thalach", "說明": "最大心率"},
            {"欄位": "exang", "說明": "運動誘發心絞痛"},
            {"欄位": "oldpeak", "說明": "運動後 ST depression"},
            {"欄位": "slope", "說明": "ST 斜率"},
            {"欄位": "ca", "說明": "螢光顯影血管數"},
            {"欄位": "thal", "說明": "Thal 檢查"},
            {"欄位": "target", "說明": "0=無心臟病紀錄，1=心臟病風險紀錄"},
        ])
        st.dataframe(field_df, use_container_width=True)

    with tab3:
        source = st.selectbox("選擇資料", ["訓練資料", "使用者評估", "健康追蹤", "模型比較", "成本效益"])
        if source == "訓練資料":
            df = heart_df
        elif source == "使用者評估":
            df = load_dynamic_csv(USER_PREDICTION_PATH)
        elif source == "健康追蹤":
            df = load_dynamic_csv(HEALTH_LOG_PATH)
        elif source == "模型比較":
            df = model_for_dashboard_df
        else:
            df = cost_df

        if df.empty:
            st.warning("目前沒有資料。")
        else:
            c1, c2 = st.columns(2)
            c1.metric("總資料筆數", len(df))
            c2.metric("資料欄位數", df.shape[1])
            st.dataframe(df.head(100), use_container_width=True)

    with tab4:
        tables = get_tables()
        if not tables:
            st.warning("讀不到 資料庫表格。")
        else:
            selected_table = st.selectbox("選擇 資料庫表格", tables)
            table_df = read_sqlite_table(selected_table, limit=100)
            st.dataframe(table_df, use_container_width=True)


st.markdown(
    """
    <div class="footer">
    AI 智慧心血管疾病風險預測與健康管理系統｜六大模組展示版 v5.3.1｜健康風險評估原型，不取代醫師診斷
    </div>
    """,
    unsafe_allow_html=True,
)
