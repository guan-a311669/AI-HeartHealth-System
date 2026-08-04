"""
dashboard/app.py

AI 智慧心血管疾病風險預測與健康管理系統
第一階段 MVP - 第 5 步：
建立 Streamlit 儀表板雛形。

功能：
1. 顯示專案介紹與免責聲明
2. 顯示 cleaned_heart_data.csv 資料概況
3. 顯示 target 分布
4. 顯示模型比較結果
5. 顯示使用者即時預測紀錄
6. 顯示風險分級分布

執行位置：
    ~/Desktop/AI_HeartHealth_System

執行指令：
    streamlit run dashboard/app.py

注意：
本系統僅作為課程專題、健康風險評估與決策支援原型，
不取代醫師診斷、醫療處置或正式臨床判斷。
"""

from pathlib import Path

import pandas as pd
import streamlit as st


# =========================
# 1. 專案路徑設定
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_heart_data.csv"
MODEL_COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"
RISK_SCORE_REPORT_PATH = BASE_DIR / "reports" / "risk_score_report.csv"
USER_PREDICTION_PATH = BASE_DIR / "reports" / "user_prediction_result.csv"


# =========================
# 2. 頁面設定
# =========================

st.set_page_config(
    page_title="AI 智慧心血管風險預測系統",
    page_icon="❤️",
    layout="wide"
)


# =========================
# 3. 讀取資料函式
# =========================

@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame:
    """讀取 CSV 檔案。"""
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


def show_file_status(file_path: Path, file_label: str) -> None:
    """顯示檔案是否存在。"""
    if file_path.exists():
        st.success(f"{file_label} 已找到")
    else:
        st.warning(f"{file_label} 尚未找到：{file_path}")


def risk_level_to_zh(value: str) -> str:
    """將英文風險等級轉成中文。"""
    mapping = {
        "low_risk": "低風險",
        "medium_risk": "中風險",
        "high_risk": "高風險",
    }
    return mapping.get(value, value)


# =========================
# 4. 主畫面
# =========================

st.title("❤️ AI 智慧心血管疾病風險預測與健康管理系統")

st.markdown("""
本系統為課程專題 MVP，使用 Cleveland Heart Disease Dataset 建立心血管風險預測模型，
並整合使用者即時輸入、低中高風險分級、健康建議與儀表板展示。
""")

st.info("提醒：本系統僅作為健康風險評估與決策支援原型，不取代醫師診斷、醫療處置或正式臨床判斷。")


# =========================
# 5. 側邊欄
# =========================

st.sidebar.title("系統選單")

page = st.sidebar.radio(
    "請選擇頁面",
    [
        "首頁總覽",
        "資料集檢查",
        "模型表現",
        "風險分布",
        "使用者預測紀錄",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("檔案狀態")
show_file_status(DATA_PATH, "cleaned_heart_data.csv")
show_file_status(MODEL_COMPARISON_PATH, "model_comparison.csv")
show_file_status(RISK_SCORE_REPORT_PATH, "risk_score_report.csv")
show_file_status(USER_PREDICTION_PATH, "user_prediction_result.csv")


# =========================
# 6. 載入資料
# =========================

heart_df = load_csv(DATA_PATH)
model_df = load_csv(MODEL_COMPARISON_PATH)
risk_df = load_csv(RISK_SCORE_REPORT_PATH)
user_df = load_csv(USER_PREDICTION_PATH)


# =========================
# 7. 首頁總覽
# =========================

if page == "首頁總覽":
    st.header("首頁總覽")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not heart_df.empty:
            st.metric("資料筆數", f"{len(heart_df)}")
        else:
            st.metric("資料筆數", "尚無資料")

    with col2:
        if not heart_df.empty:
            st.metric("資料欄位數", f"{heart_df.shape[1]}")
        else:
            st.metric("資料欄位數", "尚無資料")

    with col3:
        if not user_df.empty:
            st.metric("使用者預測筆數", f"{len(user_df)}")
        else:
            st.metric("使用者預測筆數", "0")

    with col4:
        if not model_df.empty:
            best_model = model_df.sort_values(by="auc", ascending=False).iloc[0]["model_name"]
            st.metric("目前最佳模型", best_model)
        else:
            st.metric("目前最佳模型", "尚未訓練")

    st.markdown("---")

    st.subheader("MVP 目前完成內容")
    st.write("✅ Excel 匯入 CSV 與 SQLite")
    st.write("✅ 資料庫與資料品質檢查")
    st.write("✅ AI 風險預測模型訓練")
    st.write("✅ 使用者即時輸入預測")
    st.write("✅ Streamlit 儀表板雛形")

    st.subheader("系統定位")
    st.write("""
    本 MVP 不是單純的分類模型，而是一個健康管理決策支援系統雛形。
    模型輸出的 risk_score 與 risk_level 主要用於風險分級、健康建議與後續追蹤管理。
    """)


# =========================
# 8. 資料集檢查
# =========================

elif page == "資料集檢查":
    st.header("資料集檢查")

    if heart_df.empty:
        st.error("找不到 cleaned_heart_data.csv，請先執行 01_import_excel_to_database.py。")
    else:
        st.subheader("資料基本資訊")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("資料筆數", len(heart_df))
        with col2:
            st.metric("欄位數", heart_df.shape[1])

        st.subheader("target 分布")

        if "target" in heart_df.columns:
            target_counts = heart_df["target"].value_counts().sort_index()
            target_display = pd.DataFrame({
                "target": target_counts.index,
                "count": target_counts.values
            })
            st.dataframe(target_display, use_container_width=True)
            st.bar_chart(target_display.set_index("target"))
        else:
            st.warning("資料中找不到 target 欄位。")

        st.subheader("前 10 筆資料")
        st.dataframe(heart_df.head(10), use_container_width=True)

        st.subheader("缺失值統計")
        missing_df = pd.DataFrame({
            "column": heart_df.columns,
            "missing_count": heart_df.isna().sum().values,
            "missing_rate": heart_df.isna().mean().values
        })
        st.dataframe(missing_df, use_container_width=True)


# =========================
# 9. 模型表現
# =========================

elif page == "模型表現":
    st.header("模型表現")

    if model_df.empty:
        st.error("找不到 model_comparison.csv，請先執行 03_train_model.py。")
    else:
        st.subheader("模型比較表")
        st.dataframe(model_df, use_container_width=True)

        if "auc" in model_df.columns:
            best_model = model_df.sort_values(by="auc", ascending=False).iloc[0]

            st.subheader("目前最佳模型")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("模型", best_model["model_name"])
            with col2:
                st.metric("Accuracy", f"{best_model['accuracy']:.4f}")
            with col3:
                st.metric("Recall", f"{best_model['recall']:.4f}")
            with col4:
                st.metric("F1", f"{best_model['f1']:.4f}")
            with col5:
                st.metric("AUC", f"{best_model['auc']:.4f}")

            st.subheader("AUC 比較")
            chart_df = model_df[["model_name", "auc"]].set_index("model_name")
            st.bar_chart(chart_df)

        st.warning("模型表現僅代表目前資料切分下的測試結果，不能直接視為臨床診斷能力。")


# =========================
# 10. 風險分布
# =========================

elif page == "風險分布":
    st.header("風險分布")

    if risk_df.empty:
        st.error("找不到 risk_score_report.csv，請先執行 03_train_model.py。")
    else:
        if "risk_level" in risk_df.columns:
            risk_df_display = risk_df.copy()
            risk_df_display["risk_level_zh"] = risk_df_display["risk_level"].apply(risk_level_to_zh)

            risk_counts = risk_df_display["risk_level_zh"].value_counts()
            risk_count_df = pd.DataFrame({
                "risk_level": risk_counts.index,
                "count": risk_counts.values
            })

            st.subheader("低 / 中 / 高風險分布")
            st.dataframe(risk_count_df, use_container_width=True)
            st.bar_chart(risk_count_df.set_index("risk_level"))

        if "risk_score" in risk_df.columns:
            st.subheader("風險分數統計")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("平均 risk_score", f"{risk_df['risk_score'].mean():.4f}")
            with col2:
                st.metric("最低 risk_score", f"{risk_df['risk_score'].min():.4f}")
            with col3:
                st.metric("最高 risk_score", f"{risk_df['risk_score'].max():.4f}")

            st.subheader("risk_score 前 10 筆")
            display_columns = []
            for column in ["patient_id", "age", "sex", "target", "risk_score", "risk_level"]:
                if column in risk_df.columns:
                    display_columns.append(column)

            st.dataframe(risk_df[display_columns].head(10), use_container_width=True)


# =========================
# 11. 使用者預測紀錄
# =========================

elif page == "使用者預測紀錄":
    st.header("使用者預測紀錄")

    if user_df.empty:
        st.warning("目前尚無使用者預測紀錄。請先執行 04_user_prediction.py。")
    else:
        st.subheader("使用者預測總覽")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("預測紀錄筆數", len(user_df))

        with col2:
            if "risk_score" in user_df.columns:
                st.metric("平均 risk_score", f"{user_df['risk_score'].mean():.4f}")

        if "risk_level_zh" in user_df.columns:
            st.subheader("使用者風險等級分布")
            user_risk_counts = user_df["risk_level_zh"].value_counts()
            user_risk_df = pd.DataFrame({
                "risk_level": user_risk_counts.index,
                "count": user_risk_counts.values
            })
            st.dataframe(user_risk_df, use_container_width=True)
            st.bar_chart(user_risk_df.set_index("risk_level"))

        st.subheader("最近 10 筆使用者預測")
        st.dataframe(user_df.tail(10), use_container_width=True)

        if "suggestion_text" in user_df.columns:
            st.subheader("最新一筆健康建議")
            latest_record = user_df.tail(1).iloc[0]
            st.write(latest_record["suggestion_text"])


# =========================
# 12. 頁尾
# =========================

st.markdown("---")
st.caption("AI 智慧心血管疾病風險預測與健康管理系統｜課程專題 MVP｜不取代醫師診斷")
