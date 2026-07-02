"""
dashboard/app.py

AI 智慧心血管疾病風險預測與健康管理系統
Streamlit 混合版 MVP 儀表板

版本定位：
C 混合版
1. 首頁：系統介紹與使用流程
2. 使用者風險評估：網頁輸入資料並即時預測
3. 健康建議與追蹤：顯示建議、可新增健康追蹤紀錄
4. 管理者儀表板：查看預測紀錄、風險分布、模型成果
5. 模型與資料說明：資料來源、欄位說明與系統限制

執行位置：
    ~/Desktop/AI_HeartHealth_System

執行指令：
    streamlit run dashboard/app.py

注意：
本系統僅作為課程專題、健康風險評估與決策支援原型，
不取代醫師診斷、醫療處置或正式臨床判斷。
"""

from pathlib import Path
from datetime import datetime
import json
import sqlite3

import joblib
import pandas as pd
import streamlit as st


# =========================
# 1. 專案路徑設定
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

USER_INPUT_TABLE = "user_input_records"
PREDICTION_TABLE = "model_prediction_result"
HEALTH_LOG_TABLE = "health_management_logs"


# =========================
# 2. 頁面設定
# =========================

st.set_page_config(
    page_title="AI 智慧心血管健康管理系統",
    page_icon="❤️",
    layout="wide"
)


# =========================
# 3. 資料讀取與模型載入
# =========================

@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame:
    """讀取 CSV，若檔案不存在則回傳空資料表。"""
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    """載入訓練完成的模型。"""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_feature_schema() -> dict:
    """載入模型欄位設定。"""
    if FEATURE_SCHEMA_PATH.exists():
        with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def file_status_text(file_path: Path) -> str:
    """回傳檔案狀態文字。"""
    if file_path.exists():
        return "✅ 已建立"
    return "⚠️ 尚未建立"


# =========================
# 4. 風險與建議函式
# =========================

def risk_level_from_score(score: float) -> str:
    """依 risk_score 轉換為低、中、高風險。"""
    if score < 0.30:
        return "low_risk"
    elif score < 0.70:
        return "medium_risk"
    else:
        return "high_risk"


def risk_level_zh(risk_level: str) -> str:
    """風險等級中文化。"""
    mapping = {
        "low_risk": "低風險",
        "medium_risk": "中風險",
        "high_risk": "高風險",
    }
    return mapping.get(risk_level, risk_level)


def risk_explain_text(risk_level: str) -> str:
    """風險等級說明。"""
    if risk_level == "low_risk":
        return "目前模型評估為低風險，建議維持健康生活型態並定期追蹤。"
    if risk_level == "medium_risk":
        return "目前模型評估為中風險，建議持續追蹤血壓、血脂與血糖，並視情況安排健康檢查。"
    return "目前模型評估為高風險，建議優先諮詢醫療人員，進一步評估心血管相關風險。"


def build_suggestions(input_data: dict, risk_score: float, risk_level: str) -> list:
    """依風險等級與輸入資料產生健康建議。"""
    suggestions = [risk_explain_text(risk_level)]

    if input_data.get("trestbps", 0) >= 140:
        suggestions.append("靜息血壓偏高，建議規律量測血壓並記錄趨勢。")

    if input_data.get("chol", 0) >= 240:
        suggestions.append("膽固醇數值偏高，建議追蹤血脂，並留意飲食調整與回診建議。")

    if input_data.get("fbs", 0) == 1:
        suggestions.append("空腹血糖指標異常，建議追蹤血糖與代謝相關風險。")

    if input_data.get("exang", 0) == 1:
        suggestions.append("有運動誘發心絞痛訊號，建議避免忽視症狀，必要時尋求醫療評估。")

    if input_data.get("oldpeak", 0) >= 2:
        suggestions.append("oldpeak 數值較高，建議搭配心電圖或醫療人員判讀進一步評估。")

    if input_data.get("cp", 4) in [1, 2]:
        suggestions.append("胸痛型態具有參考意義，若胸痛明顯、反覆或伴隨冒冷汗、喘、胸悶，建議儘快就醫評估。")

    suggestions.append("提醒：本結果僅供健康風險評估與課程專題展示，不代表正式診斷。")

    return suggestions


def get_risk_card_message(risk_level: str) -> tuple:
    """依風險等級回傳 Streamlit 顯示方式。"""
    if risk_level == "low_risk":
        return ("success", "低風險：維持追蹤與健康生活型態")
    if risk_level == "medium_risk":
        return ("warning", "中風險：建議追蹤並視情況安排檢查")
    return ("error", "高風險：建議優先諮詢醫療人員")


# =========================
# 5. 資料儲存函式
# =========================

def create_database_tables() -> None:
    """建立使用者輸入、預測結果與健康追蹤資料表。"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {USER_INPUT_TABLE} (
            record_id TEXT PRIMARY KEY,
            created_at TEXT,
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
            thal REAL
        )
        """)

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
            prediction_id TEXT PRIMARY KEY,
            record_id TEXT,
            created_at TEXT,
            risk_score REAL,
            risk_level TEXT,
            risk_level_zh TEXT,
            suggestion_text TEXT
        )
        """)

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {HEALTH_LOG_TABLE} (
            log_id TEXT PRIMARY KEY,
            created_at TEXT,
            systolic_bp REAL,
            diastolic_bp REAL,
            blood_sugar REAL,
            weight REAL,
            exercise_minutes REAL,
            sleep_hours REAL,
            symptom_note TEXT
        )
        """)

        conn.commit()


def save_prediction(input_data: dict, risk_score: float, risk_level: str, suggestions: list) -> str:
    """將單次預測結果存成 CSV 與 SQLite。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    create_database_tables()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_id = "WEB_" + datetime.now().strftime("%Y%m%d%H%M%S")
    prediction_id = "PRED_" + record_id
    suggestion_text = "；".join(suggestions)

    result = {
        "record_id": record_id,
        "created_at": created_at,
        **input_data,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_level_zh": risk_level_zh(risk_level),
        "suggestion_text": suggestion_text,
    }

    result_df = pd.DataFrame([result])

    if USER_PREDICTION_PATH.exists():
        old_df = pd.read_csv(USER_PREDICTION_PATH)
        new_df = pd.concat([old_df, result_df], ignore_index=True)
    else:
        new_df = result_df

    new_df.to_csv(USER_PREDICTION_PATH, index=False, encoding="utf-8-sig")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(f"""
        INSERT INTO {USER_INPUT_TABLE} (
            record_id, created_at,
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id,
            created_at,
            input_data["age"],
            input_data["sex"],
            input_data["cp"],
            input_data["trestbps"],
            input_data["chol"],
            input_data["fbs"],
            input_data["restecg"],
            input_data["thalach"],
            input_data["exang"],
            input_data["oldpeak"],
            input_data["slope"],
            input_data["ca"],
            input_data["thal"],
        ))

        cursor.execute(f"""
        INSERT INTO {PREDICTION_TABLE} (
            prediction_id, record_id, created_at,
            risk_score, risk_level, risk_level_zh, suggestion_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            record_id,
            created_at,
            risk_score,
            risk_level,
            risk_level_zh(risk_level),
            suggestion_text,
        ))

        conn.commit()

    return record_id


def save_health_log(log_data: dict) -> str:
    """儲存健康追蹤紀錄。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    create_database_tables()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_id = "HL_" + datetime.now().strftime("%Y%m%d%H%M%S")

    result = {
        "log_id": log_id,
        "created_at": created_at,
        **log_data
    }

    result_df = pd.DataFrame([result])

    if HEALTH_LOG_PATH.exists():
        old_df = pd.read_csv(HEALTH_LOG_PATH)
        new_df = pd.concat([old_df, result_df], ignore_index=True)
    else:
        new_df = result_df

    new_df.to_csv(HEALTH_LOG_PATH, index=False, encoding="utf-8-sig")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(f"""
        INSERT INTO {HEALTH_LOG_TABLE} (
            log_id, created_at,
            systolic_bp, diastolic_bp, blood_sugar, weight,
            exercise_minutes, sleep_hours, symptom_note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_id,
            created_at,
            log_data["systolic_bp"],
            log_data["diastolic_bp"],
            log_data["blood_sugar"],
            log_data["weight"],
            log_data["exercise_minutes"],
            log_data["sleep_hours"],
            log_data["symptom_note"],
        ))

        conn.commit()

    return log_id


# =========================
# 6. 載入主要資料
# =========================

heart_df = load_csv(DATA_PATH)
model_df = load_csv(MODEL_COMPARISON_PATH)
risk_df = load_csv(RISK_SCORE_REPORT_PATH)
user_df = load_csv(USER_PREDICTION_PATH)
health_log_df = load_csv(HEALTH_LOG_PATH)
model = load_model()
schema = load_feature_schema()


# =========================
# 7. 側邊欄
# =========================

st.sidebar.title("❤️ 系統選單")

page = st.sidebar.radio(
    "請選擇功能",
    [
        "首頁",
        "使用者風險評估",
        "健康建議與追蹤",
        "管理者儀表板",
        "模型與資料說明",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("MVP 檔案狀態")
st.sidebar.write(f"資料集：{file_status_text(DATA_PATH)}")
st.sidebar.write(f"模型：{file_status_text(MODEL_PATH)}")
st.sidebar.write(f"模型設定：{file_status_text(FEATURE_SCHEMA_PATH)}")
st.sidebar.write(f"模型比較：{file_status_text(MODEL_COMPARISON_PATH)}")
st.sidebar.write(f"使用者預測：{file_status_text(USER_PREDICTION_PATH)}")


# =========================
# 8. 首頁
# =========================

if page == "首頁":
    st.title("AI 智慧心血管疾病風險預測與健康管理系統")

    st.info("本系統為課程專題 MVP，僅作為健康風險評估與決策支援原型，不取代醫師診斷、醫療處置或正式臨床判斷。")

    st.subheader("系統流程")

    st.markdown("""
    **使用者流程：**  
    填寫基本與檢查資料 → AI 模型產生 risk_score → 顯示低 / 中 / 高風險 → 產生健康建議 → 進行健康追蹤

    **管理者流程：**  
    查看資料集狀態 → 查看模型表現 → 查看使用者預測紀錄 → 觀察低中高風險分布
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("訓練資料筆數", len(heart_df) if not heart_df.empty else 0)

    with col2:
        st.metric("模型比較數", len(model_df) if not model_df.empty else 0)

    with col3:
        st.metric("使用者預測筆數", len(user_df) if not user_df.empty else 0)

    with col4:
        st.metric("健康追蹤筆數", len(health_log_df) if not health_log_df.empty else 0)

    st.markdown("---")

    st.subheader("目前 MVP 已完成")
    st.write("✅ Excel 匯入 CSV 與 SQLite")
    st.write("✅ 資料品質檢查")
    st.write("✅ AI 風險預測模型訓練")
    st.write("✅ 使用者即時預測")
    st.write("✅ Streamlit 混合版儀表板")


# =========================
# 9. 使用者風險評估
# =========================

elif page == "使用者風險評估":
    st.title("使用者風險評估")

    st.warning("請依照目前可取得的資料填寫。若不確定檢查數值，建議由醫療人員協助判讀。")

    if model is None or not schema:
        st.error("找不到模型或 feature_schema.json，請先執行 03_train_model.py。")
    else:
        st.subheader("請輸入評估資料")

        with st.form("risk_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("年齡 age", min_value=1, max_value=120, value=63)
                sex_text = st.selectbox("性別 sex", ["男性", "女性"])
                cp_text = st.selectbox(
                    "胸痛型態 cp",
                    [
                        "1 典型心絞痛",
                        "2 非典型心絞痛",
                        "3 非心絞痛疼痛",
                        "4 無症狀",
                    ],
                    index=0
                )
                trestbps = st.number_input("靜息血壓 trestbps", min_value=1, max_value=260, value=145)

            with col2:
                chol = st.number_input("膽固醇 chol", min_value=1, max_value=700, value=233)
                fbs_text = st.selectbox("空腹血糖 > 120 mg/dl？ fbs", ["否 0", "是 1"])
                restecg_text = st.selectbox(
                    "靜息心電圖 restecg",
                    [
                        "0 正常",
                        "1 ST-T 異常",
                        "2 左心室肥大可能",
                    ],
                    index=2
                )
                thalach = st.number_input("最大心率 thalach", min_value=1, max_value=250, value=150)

            with col3:
                exang_text = st.selectbox("運動誘發心絞痛 exang", ["否 0", "是 1"])
                oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
                slope_text = st.selectbox("ST 斜率 slope", ["1 上升", "2 平坦", "3 下降"], index=2)
                ca = st.selectbox("顯影血管數 ca", [0, 1, 2, 3])
                thal_text = st.selectbox("thal", ["3 正常", "6 固定缺陷", "7 可逆缺陷"], index=1)

            submitted = st.form_submit_button("開始 AI 風險預測")

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

            feature_columns = schema["feature_columns"]
            input_df = pd.DataFrame([input_data], columns=feature_columns)

            risk_score = float(model.predict_proba(input_df)[:, 1][0])
            risk_level = risk_level_from_score(risk_score)
            suggestions = build_suggestions(input_data, risk_score, risk_level)
            record_id = save_prediction(input_data, risk_score, risk_level, suggestions)

            st.markdown("---")
            st.subheader("AI 預測結果")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("紀錄編號", record_id)

            with col2:
                st.metric("risk_score", f"{risk_score:.4f}")

            with col3:
                st.metric("risk_level", risk_level_zh(risk_level))

            message_type, message_text = get_risk_card_message(risk_level)

            if message_type == "success":
                st.success(message_text)
            elif message_type == "warning":
                st.warning(message_text)
            else:
                st.error(message_text)

            st.subheader("個人化健康建議")
            for index, suggestion in enumerate(suggestions, start=1):
                st.write(f"{index}. {suggestion}")

            st.info("結果已存入 reports/user_prediction_result.csv 與 SQLite 資料庫。")


# =========================
# 10. 健康建議與追蹤
# =========================

elif page == "健康建議與追蹤":
    st.title("健康建議與追蹤")

    st.info("健康追蹤資料不直接加入 Cleveland 模型訓練，而是用於趨勢觀察、異常提醒與健康管理建議。")

    st.subheader("新增健康追蹤紀錄")

    with st.form("health_log_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            systolic_bp = st.number_input("收縮壓 systolic_bp", min_value=50, max_value=260, value=120)
            diastolic_bp = st.number_input("舒張壓 diastolic_bp", min_value=30, max_value=160, value=80)

        with col2:
            blood_sugar = st.number_input("血糖 blood_sugar，可先填 0 表示未知", min_value=0, max_value=500, value=0)
            weight = st.number_input("體重 weight", min_value=1.0, max_value=300.0, value=70.0, step=0.1)

        with col3:
            exercise_minutes = st.number_input("今日運動分鐘數", min_value=0, max_value=300, value=30)
            sleep_hours = st.number_input("睡眠時數", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

        symptom_note = st.text_area("症狀或備註，例如胸悶、胸痛、喘、心悸，可留空")

        submitted_log = st.form_submit_button("儲存健康追蹤紀錄")

    if submitted_log:
        log_data = {
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "blood_sugar": float(blood_sugar),
            "weight": float(weight),
            "exercise_minutes": float(exercise_minutes),
            "sleep_hours": float(sleep_hours),
            "symptom_note": symptom_note,
        }

        log_id = save_health_log(log_data)
        st.success(f"健康追蹤紀錄已儲存：{log_id}")
        st.info("若要立即看到最新紀錄，請重新整理頁面。")

    st.markdown("---")

    st.subheader("健康追蹤紀錄")

    health_log_df = load_csv(HEALTH_LOG_PATH)

    if health_log_df.empty:
        st.warning("目前尚無健康追蹤紀錄。")
    else:
        st.dataframe(health_log_df.tail(10), use_container_width=True)

        chart_columns = []
        for column in ["systolic_bp", "diastolic_bp", "blood_sugar", "weight", "exercise_minutes", "sleep_hours"]:
            if column in health_log_df.columns:
                chart_columns.append(column)

        if chart_columns:
            st.subheader("健康趨勢圖")
            chart_df = health_log_df[chart_columns].copy()
            st.line_chart(chart_df)


# =========================
# 11. 管理者儀表板
# =========================

elif page == "管理者儀表板":
    st.title("管理者儀表板")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("訓練資料筆數", len(heart_df) if not heart_df.empty else 0)

    with col2:
        st.metric("使用者預測筆數", len(user_df) if not user_df.empty else 0)

    with col3:
        if not user_df.empty and "risk_score" in user_df.columns:
            st.metric("平均使用者 risk_score", f"{user_df['risk_score'].mean():.4f}")
        else:
            st.metric("平均使用者 risk_score", "尚無資料")

    with col4:
        if not user_df.empty and "risk_level" in user_df.columns:
            high_count = (user_df["risk_level"] == "high_risk").sum()
            st.metric("高風險筆數", high_count)
        else:
            st.metric("高風險筆數", 0)

    st.markdown("---")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("模型比較")
        if model_df.empty:
            st.warning("尚無 model_comparison.csv")
        else:
            st.dataframe(model_df, use_container_width=True)
            if "auc" in model_df.columns:
                chart_df = model_df[["model_name", "auc"]].set_index("model_name")
                st.bar_chart(chart_df)

    with right_col:
        st.subheader("使用者風險分布")
        if user_df.empty:
            st.warning("尚無使用者預測紀錄")
        elif "risk_level_zh" in user_df.columns:
            risk_counts = user_df["risk_level_zh"].value_counts()
            risk_count_df = pd.DataFrame({
                "risk_level": risk_counts.index,
                "count": risk_counts.values
            })
            st.dataframe(risk_count_df, use_container_width=True)
            st.bar_chart(risk_count_df.set_index("risk_level"))

    st.markdown("---")

    st.subheader("最近使用者預測紀錄")
    if user_df.empty:
        st.warning("尚無使用者預測紀錄，請先到『使用者風險評估』頁新增。")
    else:
        st.dataframe(user_df.tail(10), use_container_width=True)


# =========================
# 12. 模型與資料說明
# =========================

elif page == "模型與資料說明":
    st.title("模型與資料說明")

    st.subheader("資料來源")
    st.write("""
    MVP 階段使用 Cleveland Heart Disease Dataset 的清理後資料作為模型訓練資料。
    使用者自行輸入資料用於即時預測；健康追蹤資料用於趨勢圖、異常提醒與健康管理建議。
    """)

    st.subheader("模型欄位")
    if schema and "feature_columns" in schema:
        feature_df = pd.DataFrame({
            "feature": schema["feature_columns"]
        })
        st.dataframe(feature_df, use_container_width=True)
    else:
        st.warning("找不到 feature_schema.json。")

    st.subheader("資料集預覽")
    if heart_df.empty:
        st.warning("找不到 cleaned_heart_data.csv。")
    else:
        st.write(f"資料筆數：{len(heart_df)}")
        st.write(f"欄位數：{heart_df.shape[1]}")
        st.dataframe(heart_df.head(10), use_container_width=True)

    st.subheader("模型限制")
    st.write("""
    1. 本模型來自課程專題資料集，樣本數有限。
    2. 預測結果代表風險分級參考，不代表正式診斷。
    3. 健康追蹤資料目前不直接加入 Cleveland 模型訓練。
    4. 若有胸痛、胸悶、喘、冒冷汗、暈厥等症狀，應優先尋求醫療人員協助。
    """)


# =========================
# 13. 頁尾
# =========================

st.markdown("---")
st.caption("AI 智慧心血管疾病風險預測與健康管理系統｜Streamlit 混合版 MVP｜不取代醫師診斷")
