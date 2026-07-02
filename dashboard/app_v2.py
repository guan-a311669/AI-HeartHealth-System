"""
dashboard/app_v2.py
AI 智慧心血管疾病風險預測與健康管理系統｜MVP v2

執行：python -m streamlit run dashboard/app_v2.py
定位：課程專題與健康風險評估原型，不取代醫師診斷。
"""

from pathlib import Path
from datetime import datetime
import json
import sqlite3

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_heart_data.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_risk_model.pkl"
SCHEMA_PATH = BASE_DIR / "models" / "feature_schema.json"
DB_PATH = BASE_DIR / "database" / "heart_disease_project.db"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_REPORT = REPORTS_DIR / "model_comparison.csv"
RISK_REPORT = REPORTS_DIR / "risk_score_report.csv"
USER_RESULT = REPORTS_DIR / "user_prediction_result.csv"
HEALTH_LOG = REPORTS_DIR / "health_management_logs.csv"

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

COLUMN_INFO = {
    "age": "年齡",
    "sex": "性別：1=男性，0=女性",
    "cp": "胸痛型態：1典型心絞痛、2非典型心絞痛、3非心絞痛疼痛、4無症狀",
    "trestbps": "靜息血壓，單位 mmHg",
    "chol": "膽固醇，單位 mg/dl",
    "fbs": "空腹血糖是否 >120 mg/dl：1=是，0=否",
    "restecg": "靜息心電圖：0正常、1 ST-T異常、2左心室肥大可能",
    "thalach": "最大心率",
    "exang": "運動誘發心絞痛：1=是，0=否",
    "oldpeak": "運動後 ST depression",
    "slope": "ST斜率：1上升、2平坦、3下降",
    "ca": "螢光顯影血管數：0~3",
    "thal": "Thal：3正常、6固定缺陷、7可逆缺陷",
    "target": "模型訓練目標欄位：0/1，僅用於資料集標記",
    "risk_score": "AI 模型輸出的風險分數",
    "public_status": "給使用者看的目前狀態",
    "suggestion_text": "健康建議文字",
}

CP_LABEL = {1: "典型心絞痛", 2: "非典型心絞痛", 3: "非心絞痛疼痛", 4: "無症狀"}
RESTECG_LABEL = {0: "正常", 1: "ST-T 異常", 2: "左心室肥大可能"}
SLOPE_LABEL = {1: "上升", 2: "平坦", 3: "下降"}
THAL_LABEL = {3: "正常", 6: "固定缺陷", 7: "可逆缺陷"}

st.set_page_config(page_title="AI 心血管健康管理 MVP v2", page_icon="❤️", layout="wide")

@st.cache_data(show_spinner=False)
def load_static(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_dynamic(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_data(show_spinner=False)
def load_schema() -> dict:
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"feature_columns": FEATURES}

def risk_level(score: float) -> str:
    if score < 0.30:
        return "low_risk"
    if score < 0.70:
        return "medium_risk"
    return "high_risk"

def risk_level_zh(level: str) -> str:
    return {"low_risk": "低風險", "medium_risk": "中風險", "high_risk": "高風險"}.get(level, level)

def public_status(level: str) -> str:
    # 使用者頁面不直接顯示閾值，改用比較像產品的語言。
    return {
        "low_risk": "穩定觀察",
        "medium_risk": "建議加強追蹤",
        "high_risk": "建議優先諮詢醫療人員",
    }.get(level, "需要補充資料")

def bp_label(sys_bp: float) -> str:
    if sys_bp < 120:
        return "目前血壓紀錄較穩定"
    if sys_bp < 130:
        return "血壓建議持續觀察"
    if sys_bp < 140:
        return "血壓偏高，建議規律追蹤"
    return "血壓偏高，建議加強追蹤並視情況諮詢醫療人員"

def suggestions(data: dict, score: float, level: str) -> list[str]:
    out = []
    if level == "low_risk":
        out.append("建議維持規律作息、均衡飲食與定期健康追蹤。")
    elif level == "medium_risk":
        out.append("建議持續追蹤血壓、血脂與血糖，必要時安排健康檢查。")
    else:
        out.append("建議優先諮詢醫療人員，進一步評估心血管相關風險。")
    if data["trestbps"] >= 140:
        out.append("靜息血壓偏高，建議規律量測並記錄趨勢。")
    if data["chol"] >= 240:
        out.append("膽固醇偏高，建議追蹤血脂並留意飲食調整。")
    if data["fbs"] == 1:
        out.append("空腹血糖指標異常，建議追蹤血糖與代謝相關風險。")
    if data["exang"] == 1:
        out.append("有運動誘發心絞痛訊號，建議不要忽視症狀，必要時尋求醫療評估。")
    if data["oldpeak"] >= 2:
        out.append("oldpeak 數值較高，建議搭配心電圖或醫療人員判讀。")
    if data["restecg"] in [1, 2]:
        out.append(f"靜息心電圖標記為「{RESTECG_LABEL.get(int(data['restecg']))}」，建議由醫療人員協助判讀。")
    out.append("提醒：本結果僅供健康風險評估與課程專題展示，不代表正式診斷。")
    return out

def ensure_tables():
    if not DB_PATH.exists():
        return
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS web_prediction_records_v2 (
            record_id TEXT PRIMARY KEY, created_at TEXT, user_name TEXT,
            age REAL, sex REAL, cp REAL, trestbps REAL, chol REAL, fbs REAL,
            restecg REAL, thalach REAL, exang REAL, oldpeak REAL, slope REAL, ca REAL, thal REAL,
            risk_score REAL, risk_level TEXT, risk_level_zh TEXT, public_status TEXT, suggestion_text TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS health_management_logs_v2 (
            log_id TEXT PRIMARY KEY, created_at TEXT, user_name TEXT,
            systolic_bp REAL, diastolic_bp REAL, blood_sugar REAL, weight REAL,
            exercise_minutes REAL, sleep_hours REAL, symptom_note TEXT, bp_label TEXT
        )
        """)
        conn.commit()

def append_csv(path: Path, row: dict) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out

def save_prediction(user_name: str, data: dict, score: float, level: str, sug: list[str]) -> str:
    ensure_tables()
    record_id = "WEB_" + datetime.now().strftime("%Y%m%d%H%M%S")
    row = {
        "record_id": record_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_name": user_name,
        **data,
        "risk_score": score,
        "risk_level": level,
        "risk_level_zh": risk_level_zh(level),
        "public_status": public_status(level),
        "suggestion_text": "；".join(sug),
    }
    append_csv(USER_RESULT, row)
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            pd.DataFrame([row]).to_sql("web_prediction_records_v2", conn, if_exists="append", index=False)
    return record_id

def save_health(row: dict) -> str:
    ensure_tables()
    log_id = "HL_" + datetime.now().strftime("%Y%m%d%H%M%S")
    row = {
        "log_id": log_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **row,
        "bp_label": bp_label(float(row["systolic_bp"])),
    }
    out = append_csv(HEALTH_LOG, row)
    st.session_state["health_log_df"] = out
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            pd.DataFrame([row]).to_sql("health_management_logs_v2", conn, if_exists="append", index=False)
    return log_id

def sqlite_tables() -> list[str]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
    return df["name"].tolist()

def sqlite_preview(table: str, limit: int = 50) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit}", conn)

def simulate_future(model, input_data: dict, feature_cols: list[str]) -> pd.DataFrame:
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

    scenarios = [
        ("現在", "依目前輸入資料評估", current),
        ("30天後", "維持追蹤情境", current),
        ("60天後", "追蹤改善情境：血壓、血脂或部分指標改善", improved),
        ("90天後", "未追蹤且指標惡化情境", worsened),
    ]
    rows = []
    for t, name, d in scenarios:
        score = float(model.predict_proba(pd.DataFrame([d], columns=feature_cols))[:, 1][0])
        lvl = risk_level(score)
        rows.append({"時間": t, "情境": name, "risk_score": score, "目前狀態": public_status(lvl)})
    return pd.DataFrame(rows)

heart_df = load_static(DATA_PATH)
model_df = load_static(MODEL_REPORT)
risk_df = load_static(RISK_REPORT)
user_df = load_dynamic(USER_RESULT)
health_df = load_dynamic(HEALTH_LOG)
model = load_model()
schema = load_schema()
feature_cols = schema.get("feature_columns", FEATURES)

if "health_log_df" not in st.session_state:
    st.session_state["health_log_df"] = health_df
if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None

st.sidebar.title("❤️ AI 健康管理系統")
page = st.sidebar.radio("功能選單", ["首頁", "AI風險評估", "AI後續趨勢模擬", "健康追蹤", "資料庫導覽", "管理者分析"])
st.sidebar.markdown("---")
st.sidebar.caption("檔案狀態")
st.sidebar.write(f"資料：{'✅' if DATA_PATH.exists() else '⚠️'}")
st.sidebar.write(f"模型：{'✅' if MODEL_PATH.exists() else '⚠️'}")
st.sidebar.write(f"使用者預測：{len(user_df) if not user_df.empty else 0} 筆")
st.sidebar.write(f"健康追蹤：{len(st.session_state['health_log_df']) if not st.session_state['health_log_df'].empty else 0} 筆")

if page == "首頁":
    st.title("AI 智慧心血管疾病風險預測與健康管理系統")
    st.info("課程專題 MVP：AI 風險評估、後續情境模擬、健康追蹤與管理者分析。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練資料", len(heart_df) if not heart_df.empty else 0)
    c2.metric("模型數", len(model_df) if not model_df.empty else 0)
    c3.metric("使用者預測", len(user_df) if not user_df.empty else 0)
    c4.metric("健康追蹤", len(st.session_state["health_log_df"]) if not st.session_state["health_log_df"].empty else 0)
    st.markdown("""
    ### 使用流程
    1. 到「AI風險評估」輸入姓名與檢查資料。  
    2. 查看 AI risk_score、目前狀態與健康建議。  
    3. 到「AI後續趨勢模擬」查看可能的管理情境。  
    4. 到「健康追蹤」新增血壓、血糖、體重、運動、睡眠紀錄。  
    5. 到「資料庫導覽」查看欄位字典、熱力圖與資料表。  
    """)
    st.warning("本系統不取代醫師診斷。若有胸痛、胸悶、喘、冒冷汗、暈厥等症狀，請優先尋求醫療人員協助。")

elif page == "AI風險評估":
    st.title("AI 風險評估")
    if model is None:
        st.error("找不到模型，請先執行 03_train_model.py。")
    else:
        with st.form("predict_form"):
            user_name = st.text_input("姓名 / 暱稱", value="測試使用者")
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("年齡", 1, 120, 63)
                sex_text = st.selectbox("性別", ["男性", "女性"])
                cp_text = st.selectbox("胸痛型態", ["1 典型心絞痛", "2 非典型心絞痛", "3 非心絞痛疼痛", "4 無症狀"])
                trestbps = st.number_input("靜息血壓 mmHg", 1, 260, 145)
            with c2:
                chol = st.number_input("膽固醇 mg/dl", 1, 700, 233)
                fbs_text = st.selectbox("空腹血糖 >120 mg/dl", ["否 0", "是 1"])
                restecg_text = st.selectbox("靜息心電圖", ["0 正常", "1 ST-T異常", "2 左心室肥大可能"], index=2)
                thalach = st.number_input("最大心率", 1, 250, 150)
            with c3:
                exang_text = st.selectbox("運動誘發心絞痛", ["否 0", "是 1"])
                oldpeak = st.number_input("oldpeak", 0.0, 10.0, 2.3, step=0.1)
                slope_text = st.selectbox("ST斜率", ["1 上升", "2 平坦", "3 下降"], index=2)
                ca = st.selectbox("螢光顯影血管數 ca", [0, 1, 2, 3])
                thal_text = st.selectbox("Thal", ["3 正常", "6 固定缺陷", "7 可逆缺陷"], index=1)
            submit = st.form_submit_button("開始 AI 評估")
        if submit:
            data = {
                "age": float(age), "sex": 1 if sex_text == "男性" else 0,
                "cp": int(cp_text.split()[0]), "trestbps": float(trestbps), "chol": float(chol),
                "fbs": int(fbs_text.split()[-1]), "restecg": int(restecg_text.split()[0]),
                "thalach": float(thalach), "exang": int(exang_text.split()[-1]), "oldpeak": float(oldpeak),
                "slope": int(slope_text.split()[0]), "ca": int(ca), "thal": int(thal_text.split()[0]),
            }
            score = float(model.predict_proba(pd.DataFrame([data], columns=feature_cols))[:, 1][0])
            lvl = risk_level(score)
            sug = suggestions(data, score, lvl)
            record_id = save_prediction(user_name, data, score, lvl, sug)
            st.session_state["latest_prediction"] = {"user_name": user_name, "input_data": data, "risk_score": score, "risk_level": lvl, "suggestions": sug}
            st.subheader("AI 評估結果")
            a, b, c = st.columns(3)
            a.metric("紀錄編號", record_id)
            b.metric("risk_score", f"{score:.4f}")
            c.metric("目前狀態", public_status(lvl))
            if lvl == "low_risk": st.success(public_status(lvl))
            elif lvl == "medium_risk": st.warning(public_status(lvl))
            else: st.error(public_status(lvl))
            st.subheader("下一步建議")
            for i, s in enumerate(sug, 1):
                st.write(f"{i}. {s}")
            mark = pd.DataFrame([
                {"項目": "胸痛型態", "輸入值": data["cp"], "標記": CP_LABEL.get(data["cp"])},
                {"項目": "靜息心電圖", "輸入值": data["restecg"], "標記": RESTECG_LABEL.get(data["restecg"])},
                {"項目": "靜息血壓", "輸入值": data["trestbps"], "標記": bp_label(data["trestbps"])},
                {"項目": "ST斜率", "輸入值": data["slope"], "標記": SLOPE_LABEL.get(data["slope"])},
                {"項目": "Thal", "輸入值": data["thal"], "標記": THAL_LABEL.get(data["thal"])},
            ])
            st.dataframe(mark, use_container_width=True)

elif page == "AI後續趨勢模擬":
    st.title("AI 後續趨勢模擬")
    st.info("這是情境模擬，不是疾病進展預言，也不是正式診斷。")
    latest = st.session_state.get("latest_prediction")
    if latest and model:
        st.subheader(f"{latest['user_name']} 的後續情境")
        future = simulate_future(model, latest["input_data"], feature_cols)
        st.dataframe(future, use_container_width=True)
        st.line_chart(future[["時間", "risk_score"]].set_index("時間"))
    else:
        st.warning("請先到『AI風險評估』新增一筆資料。")
        if not user_df.empty:
            cols = [c for c in ["created_at", "user_name", "risk_score", "public_status", "suggestion_text"] if c in user_df.columns]
            st.dataframe(user_df.tail(10)[cols], use_container_width=True)

elif page == "健康追蹤":
    st.title("健康追蹤")
    st.info("新增後會立即更新下方表格與趨勢圖，不需要手動重整頁面。")
    latest = st.session_state.get("latest_prediction")
    default_name = latest["user_name"] if latest else "測試使用者"
    with st.form("health_form"):
        user_name = st.text_input("姓名 / 暱稱", value=default_name)
        c1, c2, c3 = st.columns(3)
        with c1:
            sys_bp = st.number_input("收縮壓", 50, 260, 120)
            dia_bp = st.number_input("舒張壓", 30, 160, 80)
        with c2:
            sugar = st.number_input("血糖，未知填0", 0, 500, 0)
            weight = st.number_input("體重 kg", 1.0, 300.0, 70.0, step=0.1)
        with c3:
            exercise = st.number_input("運動分鐘數", 0, 300, 30)
            sleep = st.number_input("睡眠時數", 0.0, 24.0, 7.0, step=0.5)
        note = st.text_area("症狀或備註")
        submit = st.form_submit_button("儲存健康追蹤")
    if submit:
        log_id = save_health({
            "user_name": user_name, "systolic_bp": float(sys_bp), "diastolic_bp": float(dia_bp),
            "blood_sugar": float(sugar), "weight": float(weight), "exercise_minutes": float(exercise),
            "sleep_hours": float(sleep), "symptom_note": note,
        })
        st.success(f"已儲存：{log_id}")
        st.write(bp_label(float(sys_bp)))
    df = st.session_state["health_log_df"]
    st.subheader("健康追蹤紀錄")
    if df.empty:
        st.warning("目前尚無健康追蹤紀錄。")
    else:
        if "user_name" in df.columns:
            name = st.selectbox("查看使用者", df["user_name"].dropna().unique().tolist())
            show = df[df["user_name"] == name].copy()
        else:
            show = df.copy()
        st.dataframe(show.tail(20), use_container_width=True)
        chart_cols = [c for c in ["systolic_bp", "diastolic_bp", "blood_sugar", "weight", "exercise_minutes", "sleep_hours"] if c in show.columns]
        if chart_cols:
            st.subheader("趨勢圖")
            st.line_chart(show[chart_cols].reset_index(drop=True))
        if "bp_label" in show.columns and not show.empty:
            st.subheader("最新標記")
            st.write(show.tail(1).iloc[0]["bp_label"])

elif page == "資料庫導覽":
    st.title("資料庫導覽")
    tabs = st.tabs(["資料產物", "欄位字典", "資料預覽", "熱力圖"])
    with tabs[0]:
        artifacts = pd.DataFrame([
            {"類型": "CSV", "名稱": "cleaned_heart_data.csv", "用途": "模型訓練主資料"},
            {"類型": "SQLite", "名稱": "heart_disease_project.db", "用途": "資料表與預測紀錄"},
            {"類型": "模型", "名稱": "heart_risk_model.pkl", "用途": "AI 即時評估"},
            {"類型": "報表", "名稱": "model_comparison.csv", "用途": "模型比較"},
            {"類型": "報表", "名稱": "user_prediction_result.csv", "用途": "使用者預測紀錄"},
            {"類型": "報表", "名稱": "health_management_logs.csv", "用途": "健康追蹤紀錄"},
        ])
        st.dataframe(artifacts, use_container_width=True)
        st.write("SQLite 資料表：", sqlite_tables())
    with tabs[1]:
        st.dataframe(pd.DataFrame([{"欄位": k, "說明": v} for k, v in COLUMN_INFO.items()]), use_container_width=True)
        labels = pd.DataFrame([
            {"欄位": "restecg", "數值": k, "意思": v} for k, v in RESTECG_LABEL.items()
        ] + [
            {"欄位": "cp", "數值": k, "意思": v} for k, v in CP_LABEL.items()
        ] + [
            {"欄位": "thal", "數值": k, "意思": v} for k, v in THAL_LABEL.items()
        ])
        st.subheader("重要欄位標記")
        st.dataframe(labels, use_container_width=True)
    with tabs[2]:
        source = st.selectbox("選擇資料", ["訓練資料", "使用者預測", "健康追蹤", "SQLite資料表"])
        if source == "訓練資料": df = heart_df
        elif source == "使用者預測": df = load_dynamic(USER_RESULT)
        elif source == "健康追蹤": df = load_dynamic(HEALTH_LOG)
        else:
            tables = sqlite_tables()
            table = st.selectbox("資料表", tables) if tables else None
            df = sqlite_preview(table) if table else pd.DataFrame()
        if df.empty:
            st.warning("目前沒有資料。")
        else:
            st.write(f"資料筆數：{len(df)}，欄位數：{df.shape[1]}")
            st.dataframe(df.head(50), use_container_width=True)
    with tabs[3]:
        if heart_df.empty:
            st.warning("找不到訓練資料。")
        else:
            num_df = heart_df.select_dtypes(include=["number"])
            default_cols = [c for c in FEATURES + ["target"] if c in num_df.columns]
            cols = st.multiselect("選擇欄位", num_df.columns.tolist(), default=default_cols)
            if len(cols) >= 2:
                corr = num_df[cols].corr()
                st.dataframe(corr.style.background_gradient(cmap="RdBu", axis=None).format("{:.2f}"), use_container_width=True)
                st.caption("熱力圖用來觀察欄位關聯，不代表因果關係。")
            else:
                st.warning("請至少選兩個欄位。")

elif page == "管理者分析":
    st.title("管理者分析")
    user_now = load_dynamic(USER_RESULT)
    a, b, c, d = st.columns(4)
    a.metric("訓練資料", len(heart_df) if not heart_df.empty else 0)
    b.metric("使用者預測", len(user_now) if not user_now.empty else 0)
    c.metric("平均risk_score", f"{user_now['risk_score'].mean():.4f}" if not user_now.empty and "risk_score" in user_now else "無")
    d.metric("健康追蹤", len(st.session_state["health_log_df"]) if not st.session_state["health_log_df"].empty else 0)
    left, right = st.columns(2)
    with left:
        st.subheader("模型比較")
        if model_df.empty:
            st.warning("尚無模型比較報表。")
        else:
            st.dataframe(model_df, use_container_width=True)
            if "auc" in model_df.columns:
                st.bar_chart(model_df[["model_name", "auc"]].set_index("model_name"))
    with right:
        st.subheader("使用者狀態分布")
        if user_now.empty:
            st.warning("尚無使用者預測。")
        else:
            col = "public_status" if "public_status" in user_now.columns else "risk_level_zh"
            counts = user_now[col].value_counts().reset_index()
            counts.columns = ["狀態", "筆數"]
            st.dataframe(counts, use_container_width=True)
            st.bar_chart(counts.set_index("狀態"))
    st.subheader("最近預測紀錄")
    if not user_now.empty:
        cols = [c for c in ["created_at", "user_name", "risk_score", "public_status", "risk_level_zh", "suggestion_text"] if c in user_now.columns]
        st.dataframe(user_now.tail(20)[cols], use_container_width=True)

st.markdown("---")
st.caption("AI 智慧心血管疾病風險預測與健康管理系統｜MVP v2｜健康風險評估原型，不取代醫師診斷")
