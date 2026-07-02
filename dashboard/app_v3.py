"""
dashboard/app_v3.py

AI 智慧心血管疾病風險預測與健康管理系統
Streamlit 美化版 v3

本版新增：
1. 美化首頁、卡片、頁面區塊。
2. AI 評估可輸入姓名，並可選擇評估日期與時間。
3. 健康追蹤可選擇紀錄日期與時間。
4. 管理者分析與健康追蹤可用日期區間篩選。
5. 健康追蹤新增後會立即更新表格與趨勢圖。
6. 資料庫導覽更清楚，含欄位說明、資料預覽與相關係數熱力圖。

執行位置：
    ~/Desktop/AI_HeartHealth_Cursor_MVP

執行指令：
    python -m streamlit run dashboard/app_v3.py

注意：
本系統僅作為課程專題、健康風險評估與決策支援原型，
不取代醫師診斷、醫療處置或正式臨床判斷。
"""

from pathlib import Path
from datetime import datetime, date, time
import json
import sqlite3

import joblib
import pandas as pd
import streamlit as st


# =========================
# 1. 路徑設定
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_heart_data.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_risk_model.pkl"
FEATURE_SCHEMA_PATH = BASE_DIR / "models" / "feature_schema.json"
DB_PATH = BASE_DIR / "database" / "heart_disease_project.db"

REPORTS_DIR = BASE_DIR / "reports"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
RISK_SCORE_REPORT_PATH = REPORTS_DIR / "risk_score_report.csv"
USER_PREDICTION_PATH = REPORTS_DIR / "user_prediction_result.csv"
HEALTH_LOG_PATH = REPORTS_DIR / "health_management_logs.csv"

WEB_PREDICTION_TABLE = "web_prediction_records_v3"
HEALTH_LOG_TABLE = "health_management_logs_v3"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


# =========================
# 2. 頁面與美化設定
# =========================

st.set_page_config(
    page_title="AI 心血管健康管理系統",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 40%, #FFFFFF 100%);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .hero-card {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 55%, #7C3AED 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(37, 99, 235, 0.22);
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .hero-subtitle {
        font-size: 17px;
        line-height: 1.7;
        opacity: 0.95;
    }

    .soft-card {
        padding: 20px 22px;
        border-radius: 20px;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .mini-note {
        color: #64748B;
        font-size: 14px;
        line-height: 1.7;
    }

    .status-stable {
        padding: 18px 20px;
        border-radius: 18px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        font-weight: 700;
    }

    .status-follow {
        padding: 18px 20px;
        border-radius: 18px;
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        font-weight: 700;
    }

    .status-priority {
        padding: 18px 20px;
        border-radius: 18px;
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        font-weight: 700;
    }

    .footer-note {
        font-size: 13px;
        color: #64748B;
        text-align: center;
        padding: 18px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 16px 18px;
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 3. 欄位說明與標記
# =========================

COLUMN_INFO = {
    "user_name": "使用者姓名或暱稱",
    "record_datetime": "使用者選擇的評估時間",
    "age": "年齡",
    "sex": "性別，1=男性，0=女性",
    "cp": "胸痛型態，1=典型心絞痛，2=非典型心絞痛，3=非心絞痛疼痛，4=無症狀",
    "trestbps": "靜息血壓，單位 mmHg",
    "chol": "膽固醇，單位 mg/dl",
    "fbs": "空腹血糖是否大於 120 mg/dl，1=是，0=否",
    "restecg": "靜息心電圖，0=正常，1=ST-T異常，2=左心室肥大可能",
    "thalach": "最大心率",
    "exang": "運動誘發心絞痛，1=是，0=否",
    "oldpeak": "運動後 ST depression",
    "slope": "ST 斜率，1=上升，2=平坦，3=下降",
    "ca": "螢光顯影血管數，0~3",
    "thal": "Thal 檢查，3=正常，6=固定缺陷，7=可逆缺陷",
    "risk_score": "模型輸出的風險分數，越高代表模型評估風險越高",
    "public_status": "系統對使用者顯示的目前狀態",
    "suggestion_text": "健康建議文字",
}

RESTECG_LABEL = {0: "正常", 1: "ST-T 異常", 2: "左心室肥大可能"}
CP_LABEL = {1: "典型心絞痛", 2: "非典型心絞痛", 3: "非心絞痛疼痛", 4: "無症狀"}
SLOPE_LABEL = {1: "上升", 2: "平坦", 3: "下降"}
THAL_LABEL = {3: "正常", 6: "固定缺陷", 7: "可逆缺陷"}


# =========================
# 4. 讀取資料與工具函式
# =========================

@st.cache_data(show_spinner=False)
def load_static_csv(path: Path) -> pd.DataFrame:
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
        with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"feature_columns": FEATURE_COLUMNS}


def combine_date_time(d: date, t: time) -> str:
    return datetime.combine(d, t).strftime("%Y-%m-%d %H:%M:%S")


def to_datetime_series(df: pd.DataFrame) -> pd.Series:
    if "record_datetime" in df.columns:
        return pd.to_datetime(df["record_datetime"], errors="coerce")
    if "created_at" in df.columns:
        return pd.to_datetime(df["created_at"], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def filter_by_date(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty:
        return df

    temp = df.copy()
    temp["_dt"] = to_datetime_series(temp)

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    temp = temp[(temp["_dt"] >= start_dt) & (temp["_dt"] <= end_dt)]
    return temp.drop(columns=["_dt"])


def risk_level_from_score(score: float) -> str:
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


def internal_risk_zh(status: str) -> str:
    mapping = {
        "stable": "低風險",
        "follow": "中風險",
        "priority": "高風險",
    }
    return mapping.get(status, status)


def status_box(status: str, text: str) -> None:
    class_name = {
        "stable": "status-stable",
        "follow": "status-follow",
        "priority": "status-priority",
    }.get(status, "status-follow")

    st.markdown(f"<div class='{class_name}'>{text}</div>", unsafe_allow_html=True)


def bp_label(systolic: float, diastolic: float | None = None) -> str:
    if systolic < 120 and (diastolic is None or diastolic < 80):
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
        suggestions.append(f"靜息心電圖標記為「{RESTECG_LABEL.get(int(input_data.get('restecg')), '未標記')}」，建議由醫療人員協助判讀。")

    if input_data.get("exang", 0) == 1:
        suggestions.append("有運動誘發心絞痛訊號，建議不要忽視症狀，必要時尋求醫療評估。")

    if input_data.get("oldpeak", 0) >= 2:
        suggestions.append("oldpeak 數值較高，建議搭配心電圖或醫療人員判讀。")

    suggestions.append("本結果僅作為健康風險評估與課程展示，不代表正式診斷。")
    return suggestions


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
        status = risk_level_from_score(score)

        result.append({
            "時間": period,
            "情境": scenario,
            "risk_score": score,
            "目前狀態": public_status_label(status),
        })

    return pd.DataFrame(result)


def create_db_tables() -> None:
    if not DB_PATH.exists():
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS web_prediction_records_v3 (
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

        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS health_management_logs_v3 (
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
    create_db_tables()

    record_id = "WEB_" + datetime.now().strftime("%Y%m%d%H%M%S")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "record_id": record_id,
        "created_at": created_at,
        "record_datetime": record_datetime,
        "user_name": user_name,
        **input_data,
        "risk_score": risk_score,
        "internal_risk_level": internal_risk_zh(status),
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
            row_df.to_sql("web_prediction_records_v3", conn, if_exists="append", index=False)

    return record_id


def save_health_log(record_datetime: str, user_name: str, log_data: dict) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    create_db_tables()

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
            row_df.to_sql("health_management_logs_v3", conn, if_exists="append", index=False)

    return log_id


def get_sqlite_tables() -> list:
    if not DB_PATH.exists():
        return []

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            conn
        )

    return df["name"].tolist()


def read_sqlite_table(table_name: str, limit: int = 50) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)


# =========================
# 5. 載入資料
# =========================

heart_df = load_static_csv(DATA_PATH)
model_df = load_static_csv(MODEL_COMPARISON_PATH)
risk_df = load_static_csv(RISK_SCORE_REPORT_PATH)
user_df = load_dynamic_csv(USER_PREDICTION_PATH)
health_df = load_dynamic_csv(HEALTH_LOG_PATH)
model = load_model()
schema = load_schema()
feature_columns = schema.get("feature_columns", FEATURE_COLUMNS)

if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None

if "health_df" not in st.session_state:
    st.session_state["health_df"] = health_df


# =========================
# 6. 側邊欄
# =========================

st.sidebar.title("❤️ AI 健康管理")
page = st.sidebar.radio(
    "功能選單",
    ["首頁", "AI 評估", "後續趨勢模擬", "健康追蹤", "資料庫導覽", "管理者分析"]
)

st.sidebar.markdown("---")
st.sidebar.caption("快速狀態")
st.sidebar.write(f"訓練資料：{len(heart_df) if not heart_df.empty else 0} 筆")
st.sidebar.write(f"使用者預測：{len(user_df) if not user_df.empty else 0} 筆")
st.sidebar.write(f"健康追蹤：{len(st.session_state['health_df']) if not st.session_state['health_df'].empty else 0} 筆")


# =========================
# 7. 首頁
# =========================

if page == "首頁":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">AI 智慧心血管疾病風險預測與健康管理系統</div>
            <div class="hero-subtitle">
                整合 AI 風險評估、後續情境模擬、健康追蹤與管理者資料分析的課程專題 MVP。
                本系統用於健康風險評估與決策支援展示，不取代醫師診斷。
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("訓練資料", len(heart_df) if not heart_df.empty else 0)
    col2.metric("模型比較", len(model_df) if not model_df.empty else 0)
    col3.metric("使用者預測", len(user_df) if not user_df.empty else 0)
    col4.metric("健康追蹤", len(st.session_state["health_df"]) if not st.session_state["health_df"].empty else 0)

    st.markdown('<div class="section-title">系統流程</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="soft-card">
        <b>使用者：</b>輸入姓名與檢查資料 → AI 評估目前狀態 → 查看後續情境模擬 → 建立健康追蹤紀錄。<br>
        <b>管理者：</b>查看模型表現 → 查看資料庫導覽 → 用熱力圖了解欄位關聯 → 追蹤使用者紀錄。
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning("若有胸痛、胸悶、喘、冒冷汗、暈厥等症狀，應優先尋求醫療人員協助。")


# =========================
# 8. AI 評估
# =========================

elif page == "AI 評估":
    st.title("AI 評估")
    st.caption("填寫姓名、時間與檢查資料後，系統會產生 AI risk_score 與目前狀態。")

    if model is None:
        st.error("找不到模型檔案，請先執行 03_train_model.py。")
    else:
        with st.form("prediction_form"):
            col_time1, col_time2, col_time3 = st.columns([2, 1, 1])
            with col_time1:
                user_name = st.text_input("姓名 / 暱稱", value="測試使用者")
            with col_time2:
                record_date = st.date_input("評估日期", value=date.today())
            with col_time3:
                record_time = st.time_input("評估時間", value=datetime.now().time().replace(second=0, microsecond=0))

            st.markdown("### 基本與檢查資料")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("年齡", min_value=1, max_value=120, value=63)
                sex_text = st.selectbox("性別", ["男性", "女性"])
                cp_text = st.selectbox(
                    "胸痛型態",
                    ["1 典型心絞痛", "2 非典型心絞痛", "3 非心絞痛疼痛", "4 無症狀"],
                    index=0
                )
                trestbps = st.number_input("靜息血壓 mmHg", min_value=1, max_value=260, value=145)

            with col2:
                chol = st.number_input("膽固醇 mg/dl", min_value=1, max_value=700, value=233)
                fbs_text = st.selectbox("空腹血糖是否 > 120 mg/dl", ["否 0", "是 1"])
                restecg_text = st.selectbox("靜息心電圖", ["0 正常", "1 ST-T 異常", "2 左心室肥大可能"], index=2)
                thalach = st.number_input("最大心率", min_value=1, max_value=250, value=150)

            with col3:
                exang_text = st.selectbox("運動誘發心絞痛", ["否 0", "是 1"])
                oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
                slope_text = st.selectbox("ST 斜率", ["1 上升", "2 平坦", "3 下降"], index=2)
                ca = st.selectbox("螢光顯影血管數 ca", [0, 1, 2, 3])
                thal_text = st.selectbox("Thal 檢查", ["3 正常", "6 固定缺陷", "7 可逆缺陷"], index=1)

            submitted = st.form_submit_button("開始 AI 評估")

        if submitted:
            input_data = {
                "age": float(age),
                "sex": 1 if sex_text == "男性" else 0,
                "cp": int(cp_text.split()[0]),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": int(fbs_text.split()[-1]),
                "restecg": int(restecg_text.split()[0]),
                "thalach": float(thalach),
                "exang": int(exang_text.split()[-1]),
                "oldpeak": float(oldpeak),
                "slope": int(slope_text.split()[0]),
                "ca": int(ca),
                "thal": int(thal_text.split()[0]),
            }

            record_datetime = combine_date_time(record_date, record_time)
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            risk_score = float(model.predict_proba(input_df)[:, 1][0])
            status = risk_level_from_score(risk_score)
            suggestions = build_suggestions(input_data, status)

            record_id = save_prediction(record_datetime, user_name, input_data, risk_score, status, suggestions)

            st.session_state["latest_prediction"] = {
                "record_id": record_id,
                "user_name": user_name,
                "record_datetime": record_datetime,
                "input_data": input_data,
                "risk_score": risk_score,
                "status": status,
                "suggestions": suggestions,
            }

            st.markdown("---")
            st.subheader("AI 評估結果")

            col1, col2, col3 = st.columns(3)
            col1.metric("姓名 / 暱稱", user_name)
            col2.metric("評估時間", record_datetime)
            col3.metric("risk_score", f"{risk_score:.4f}")

            status_box(status, public_status_label(status))

            st.markdown("### 下一步建議")
            for i, suggestion in enumerate(suggestions, start=1):
                st.write(f"{i}. {suggestion}")

            st.markdown("### 欄位標記")
            label_df = pd.DataFrame([
                {"項目": "胸痛型態", "輸入值": input_data["cp"], "標記": CP_LABEL.get(input_data["cp"], "未標記")},
                {"項目": "靜息心電圖", "輸入值": input_data["restecg"], "標記": RESTECG_LABEL.get(input_data["restecg"], "未標記")},
                {"項目": "靜息血壓", "輸入值": input_data["trestbps"], "標記": bp_label(input_data["trestbps"])},
                {"項目": "ST 斜率", "輸入值": input_data["slope"], "標記": SLOPE_LABEL.get(input_data["slope"], "未標記")},
                {"項目": "Thal", "輸入值": input_data["thal"], "標記": THAL_LABEL.get(input_data["thal"], "未標記")},
            ])
            st.dataframe(label_df, use_container_width=True)

            st.success("預測結果已儲存，可到「後續趨勢模擬」或「管理者分析」查看。")


# =========================
# 9. 後續趨勢模擬
# =========================

elif page == "後續趨勢模擬":
    st.title("後續趨勢模擬")
    st.info("這是根據目前輸入資料做的情境模擬，不是疾病進展預言，也不代表正式醫療判斷。")

    latest = st.session_state.get("latest_prediction")

    if latest is None:
        st.warning("目前沒有本次最新 AI 評估紀錄。請先到「AI 評估」新增一筆資料。")
    elif model is None:
        st.error("找不到模型檔案。")
    else:
        st.subheader(f"{latest['user_name']} 的情境模擬")
        st.caption(f"評估時間：{latest['record_datetime']}")

        future_df = simulate_future(model, feature_columns, latest["input_data"])

        st.dataframe(future_df, use_container_width=True)
        st.line_chart(future_df[["時間", "risk_score"]].set_index("時間"))

        st.markdown(
            """
            <div class="mini-note">
            說明：改善情境會假設血壓、膽固醇或部分指標改善；惡化情境會假設血壓、膽固醇或症狀訊號上升。
            這只是決策支援展示，不代表真實疾病進展。
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# 10. 健康追蹤
# =========================

elif page == "健康追蹤":
    st.title("健康追蹤")
    st.caption("健康追蹤資料用於趨勢觀察，不直接加入 Cleveland 模型訓練。")

    latest = st.session_state.get("latest_prediction")
    default_name = latest["user_name"] if latest is not None else "測試使用者"

    with st.form("health_form"):
        col_time1, col_time2, col_time3 = st.columns([2, 1, 1])
        with col_time1:
            user_name = st.text_input("姓名 / 暱稱", value=default_name)
        with col_time2:
            log_date = st.date_input("紀錄日期", value=date.today())
        with col_time3:
            log_time = st.time_input("紀錄時間", value=datetime.now().time().replace(second=0, microsecond=0))

        col1, col2, col3 = st.columns(3)

        with col1:
            systolic_bp = st.number_input("收縮壓", min_value=50, max_value=260, value=120)
            diastolic_bp = st.number_input("舒張壓", min_value=30, max_value=160, value=80)

        with col2:
            blood_sugar = st.number_input("血糖，未知可填 0", min_value=0, max_value=500, value=0)
            weight = st.number_input("體重 kg", min_value=1.0, max_value=300.0, value=70.0, step=0.1)

        with col3:
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
        st.write(bp_label(float(systolic_bp), float(diastolic_bp)))

    st.markdown("---")
    st.subheader("追蹤紀錄與趨勢")

    current_df = st.session_state["health_df"]

    if current_df.empty:
        st.warning("目前尚無健康追蹤紀錄。")
    else:
        col_filter1, col_filter2, col_filter3 = st.columns(3)

        with col_filter1:
            names = current_df["user_name"].dropna().unique().tolist() if "user_name" in current_df.columns else ["全部"]
            selected_name = st.selectbox("選擇使用者", names)

        with col_filter2:
            start_date = st.date_input("開始日期", value=date.today().replace(day=1), key="health_start")

        with col_filter3:
            end_date = st.date_input("結束日期", value=date.today(), key="health_end")

        show_df = current_df.copy()

        if "user_name" in show_df.columns:
            show_df = show_df[show_df["user_name"] == selected_name]

        show_df = filter_by_date(show_df, start_date, end_date)

        st.dataframe(show_df.tail(30), use_container_width=True)

        trend_cols = ["systolic_bp", "diastolic_bp", "blood_sugar", "weight", "exercise_minutes", "sleep_hours"]
        available_cols = [col for col in trend_cols if col in show_df.columns]

        if available_cols and not show_df.empty:
            st.line_chart(show_df[available_cols].reset_index(drop=True))

            latest_row = show_df.tail(1).iloc[0]
            if "bp_label" in show_df.columns:
                st.info(latest_row["bp_label"])


# =========================
# 11. 資料庫導覽
# =========================

elif page == "資料庫導覽":
    st.title("資料庫導覽")
    st.caption("這一頁讓分享對象可以快速理解：有哪些資料、欄位代表什麼、資料長什麼樣。")

    tab1, tab2, tab3, tab4 = st.tabs(["資料產物", "欄位字典", "資料預覽", "熱力圖"])

    with tab1:
        artifacts = pd.DataFrame([
            {"類型": "CSV", "名稱": "cleaned_heart_data.csv", "用途": "模型訓練與臨床指標分析主資料"},
            {"類型": "SQLite", "名稱": "heart_disease_project.db", "用途": "統一儲存資料表、預測結果與健康紀錄"},
            {"類型": "模型", "名稱": "heart_risk_model.pkl", "用途": "AI 即時風險評估"},
            {"類型": "報表", "名稱": "model_comparison.csv", "用途": "模型表現比較"},
            {"類型": "報表", "名稱": "risk_score_report.csv", "用途": "全資料風險分數"},
            {"類型": "報表", "名稱": "user_prediction_result.csv", "用途": "使用者預測紀錄"},
            {"類型": "報表", "名稱": "health_management_logs.csv", "用途": "健康追蹤紀錄"},
        ])
        st.dataframe(artifacts, use_container_width=True)

        st.subheader("SQLite 資料表")
        tables = get_sqlite_tables()
        if tables:
            st.write(tables)
        else:
            st.warning("目前讀不到 SQLite 資料表。")

    with tab2:
        info_df = pd.DataFrame([{"欄位": k, "說明": v} for k, v in COLUMN_INFO.items()])
        st.dataframe(info_df, use_container_width=True)

        label_df = pd.DataFrame([
            {"欄位": "restecg", "數值": 0, "意思": "正常"},
            {"欄位": "restecg", "數值": 1, "意思": "ST-T 異常"},
            {"欄位": "restecg", "數值": 2, "意思": "左心室肥大可能"},
            {"欄位": "cp", "數值": 1, "意思": "典型心絞痛"},
            {"欄位": "cp", "數值": 2, "意思": "非典型心絞痛"},
            {"欄位": "cp", "數值": 3, "意思": "非心絞痛疼痛"},
            {"欄位": "cp", "數值": 4, "意思": "無症狀"},
            {"欄位": "thal", "數值": 3, "意思": "正常"},
            {"欄位": "thal", "數值": 6, "意思": "固定缺陷"},
            {"欄位": "thal", "數值": 7, "意思": "可逆缺陷"},
        ])
        st.subheader("常見代碼標記")
        st.dataframe(label_df, use_container_width=True)

    with tab3:
        source = st.selectbox(
            "選擇資料",
            ["訓練資料", "使用者預測", "健康追蹤", "SQLite 資料表"]
        )

        if source == "訓練資料":
            df = heart_df
        elif source == "使用者預測":
            df = load_dynamic_csv(USER_PREDICTION_PATH)
        elif source == "健康追蹤":
            df = load_dynamic_csv(HEALTH_LOG_PATH)
        else:
            tables = get_sqlite_tables()
            if tables:
                selected_table = st.selectbox("選擇資料表", tables)
                df = read_sqlite_table(selected_table)
            else:
                df = pd.DataFrame()

        if df.empty:
            st.warning("目前沒有資料。")
        else:
            col1, col2 = st.columns(2)
            col1.metric("資料筆數", len(df))
            col2.metric("欄位數", df.shape[1])
            st.dataframe(df.head(50), use_container_width=True)

    with tab4:
        if heart_df.empty:
            st.warning("找不到訓練資料。")
        else:
            numeric_df = heart_df.select_dtypes(include=["number"])
            default_cols = [col for col in FEATURE_COLUMNS + ["target"] if col in numeric_df.columns]

            selected_cols = st.multiselect(
                "選擇熱力圖欄位",
                numeric_df.columns.tolist(),
                default=default_cols
            )

            if len(selected_cols) < 2:
                st.warning("請至少選擇兩個欄位。")
            else:
                corr_df = numeric_df[selected_cols].corr()

                st.dataframe(
                    corr_df.style.background_gradient(cmap="RdBu", axis=None).format("{:.2f}"),
                    use_container_width=True
                )

                st.caption("熱力圖呈現欄位間線性關聯，不代表因果關係。")


# =========================
# 12. 管理者分析
# =========================

elif page == "管理者分析":
    st.title("管理者分析")

    latest_user_df = load_dynamic_csv(USER_PREDICTION_PATH)

    if not latest_user_df.empty:
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            start_date = st.date_input("分析開始日期", value=date.today().replace(day=1), key="admin_start")
        with col_filter2:
            end_date = st.date_input("分析結束日期", value=date.today(), key="admin_end")

        latest_user_df = filter_by_date(latest_user_df, start_date, end_date)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("訓練資料筆數", len(heart_df) if not heart_df.empty else 0)
    col2.metric("區間預測筆數", len(latest_user_df) if not latest_user_df.empty else 0)

    if not latest_user_df.empty and "risk_score" in latest_user_df.columns:
        col3.metric("平均 risk_score", f"{latest_user_df['risk_score'].mean():.4f}")
    else:
        col3.metric("平均 risk_score", "尚無資料")

    if not latest_user_df.empty and "internal_risk_level" in latest_user_df.columns:
        priority_count = (latest_user_df["internal_risk_level"] == "高風險").sum()
        col4.metric("需優先關注筆數", int(priority_count))
    else:
        col4.metric("需優先關注筆數", 0)

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("模型比較")
        if model_df.empty:
            st.warning("尚無 model_comparison.csv。")
        else:
            st.dataframe(model_df, use_container_width=True)
            if "auc" in model_df.columns:
                st.bar_chart(model_df[["model_name", "auc"]].set_index("model_name"))

    with right:
        st.subheader("使用者目前狀態分布")
        if latest_user_df.empty:
            st.warning("尚無使用者預測資料。")
        else:
            status_col = "public_status" if "public_status" in latest_user_df.columns else "risk_level_zh"
            if status_col in latest_user_df.columns:
                count_df = latest_user_df[status_col].value_counts().reset_index()
                count_df.columns = ["狀態", "筆數"]
                st.dataframe(count_df, use_container_width=True)
                st.bar_chart(count_df.set_index("狀態"))

    st.markdown("---")
    st.subheader("最近預測紀錄")

    if latest_user_df.empty:
        st.warning("目前沒有資料。")
    else:
        display_cols = [
            col for col in [
                "record_datetime", "created_at", "user_name",
                "risk_score", "public_status", "internal_risk_level", "suggestion_text"
            ]
            if col in latest_user_df.columns
        ]
        st.dataframe(latest_user_df.tail(30)[display_cols], use_container_width=True)


# =========================
# 13. 頁尾
# =========================

st.markdown(
    """
    <div class="footer-note">
        AI 智慧心血管疾病風險預測與健康管理系統｜MVP v3｜健康風險評估原型，不取代醫師診斷
    </div>
    """,
    unsafe_allow_html=True
)
