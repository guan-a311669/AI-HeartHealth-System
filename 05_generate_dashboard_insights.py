"""
05_generate_dashboard_insights.py

AI 智慧心血管疾病風險預測與健康管理系統
第一階段 MVP 強化版 - 第 5-1 步：
產生「六個子專案展示版儀表板」需要的分析資料。

目的：
目前 app_v4 已經可以跑，但圖表與六個子專案展示感還不夠。
這支程式會把現有資料整理成多個 CSV，讓後續 app_v5.py 可以直接畫更多圖表。

執行位置：
    ~/Desktop/AI_HeartHealth_Cursor_MVP

執行指令：
    python 05_generate_dashboard_insights.py

輸出位置：
    reports/dashboard_insights/
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd


# =========================
# 1. 專案路徑設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_heart_data.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_risk_model.pkl"
FEATURE_SCHEMA_PATH = BASE_DIR / "models" / "feature_schema.json"

REPORTS_DIR = BASE_DIR / "reports"
USER_PREDICTION_PATH = REPORTS_DIR / "user_prediction_result.csv"
HEALTH_LOG_PATH = REPORTS_DIR / "health_management_logs.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
RISK_SCORE_REPORT_PATH = REPORTS_DIR / "risk_score_report.csv"

OUTPUT_DIR = REPORTS_DIR / "dashboard_insights"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

LABEL_MAP = {
    "sex": {0: "女性", 1: "男性"},
    "cp": {1: "典型心絞痛", 2: "非典型心絞痛", 3: "非心絞痛疼痛", 4: "無症狀"},
    "fbs": {0: "未大於120", 1: "大於120"},
    "restecg": {0: "正常", 1: "ST-T異常", 2: "左心室肥大可能"},
    "exang": {0: "否", 1: "是"},
    "slope": {1: "上升", 2: "平坦", 3: "下降"},
    "thal": {3: "正常", 6: "固定缺陷", 7: "可逆缺陷"},
}


# =========================
# 2. 通用工具函式
# =========================

def read_csv_safely(path: Path) -> pd.DataFrame:
    """安全讀取 CSV。若檔案不存在，回傳空 DataFrame。"""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """統一輸出 CSV。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已輸出：{output_path}")
    return output_path


def normalize_status(value) -> str:
    """把不同版本 app 產生的風險欄位統一成 stable / follow / priority。"""
    text = str(value)

    if "高" in text or "priority" in text or "優先" in text:
        return "priority"
    if "中" in text or "follow" in text or "加強" in text:
        return "follow"
    if "低" in text or "stable" in text or "穩定" in text:
        return "stable"

    return "unknown"


def status_zh(value) -> str:
    """將內部狀態轉中文顯示。"""
    status = normalize_status(value)

    mapping = {
        "stable": "穩定觀察",
        "follow": "建議加強追蹤",
        "priority": "建議優先關注",
        "unknown": "未分類",
    }

    return mapping.get(status, "未分類")


# =========================
# 3. 子專案一：資料庫與資料品質
# =========================

def generate_data_quality_insights(heart_df: pd.DataFrame) -> None:
    """產生資料品質、target 分布、年齡分布等摘要。"""
    if heart_df.empty:
        print("找不到 cleaned_heart_data.csv，略過子專案一資料品質摘要。")
        return

    # target 分布
    if "target" in heart_df.columns:
        target_counts = heart_df["target"].value_counts().sort_index()
        target_df = pd.DataFrame({
            "target": target_counts.index,
            "target_label": ["no_heart_disease" if x == 0 else "heart_disease_risk" for x in target_counts.index],
            "count": target_counts.values,
            "percent": (target_counts.values / len(heart_df) * 100).round(2)
        })
        save_csv(target_df, "target_distribution.csv")

    # 缺失值摘要
    missing_df = pd.DataFrame({
        "column_name": heart_df.columns,
        "missing_count": heart_df.isna().sum().values,
        "missing_rate": (heart_df.isna().mean().values * 100).round(2),
        "dtype": [str(heart_df[col].dtype) for col in heart_df.columns],
        "unique_count": [heart_df[col].nunique(dropna=True) for col in heart_df.columns],
    })
    save_csv(missing_df, "data_quality_overview.csv")

    # 年齡、血壓、膽固醇分組摘要
    group_rows = []

    if "age_group" in heart_df.columns and "target" in heart_df.columns:
        for group, sub_df in heart_df.groupby("age_group"):
            group_rows.append({
                "dimension": "age_group",
                "group": group,
                "count": len(sub_df),
                "risk_count": int(sub_df["target"].sum()),
                "risk_rate": round(sub_df["target"].mean() * 100, 2),
            })

    if "bp_group" in heart_df.columns and "target" in heart_df.columns:
        for group, sub_df in heart_df.groupby("bp_group"):
            group_rows.append({
                "dimension": "bp_group",
                "group": group,
                "count": len(sub_df),
                "risk_count": int(sub_df["target"].sum()),
                "risk_rate": round(sub_df["target"].mean() * 100, 2),
            })

    if "chol_group" in heart_df.columns and "target" in heart_df.columns:
        for group, sub_df in heart_df.groupby("chol_group"):
            group_rows.append({
                "dimension": "chol_group",
                "group": group,
                "count": len(sub_df),
                "risk_count": int(sub_df["target"].sum()),
                "risk_rate": round(sub_df["target"].mean() * 100, 2),
            })

    if group_rows:
        save_csv(pd.DataFrame(group_rows), "clinical_group_risk_summary.csv")


# =========================
# 4. 子專案二：臨床指標關聯分析
# =========================

def generate_clinical_feature_insights(heart_df: pd.DataFrame) -> None:
    """產生類別欄位風險比例、連續欄位有病/無病比較。"""
    if heart_df.empty or "target" not in heart_df.columns:
        print("資料不存在或找不到 target，略過子專案二臨床分析。")
        return

    categorical_rows = []

    for column in CATEGORICAL_COLUMNS:
        if column not in heart_df.columns:
            continue

        for value, sub_df in heart_df.groupby(column):
            label = LABEL_MAP.get(column, {}).get(int(value) if pd.notna(value) else value, str(value))

            categorical_rows.append({
                "feature": column,
                "value": value,
                "label": label,
                "sample_count": len(sub_df),
                "risk_count": int(sub_df["target"].sum()),
                "risk_rate": round(sub_df["target"].mean() * 100, 2),
            })

    if categorical_rows:
        save_csv(pd.DataFrame(categorical_rows), "clinical_categorical_risk_table.csv")

    numeric_rows = []

    for column in NUMERIC_COLUMNS:
        if column not in heart_df.columns:
            continue

        group0 = heart_df[heart_df["target"] == 0][column]
        group1 = heart_df[heart_df["target"] == 1][column]

        numeric_rows.append({
            "feature": column,
            "no_risk_mean": round(group0.mean(), 2),
            "risk_mean": round(group1.mean(), 2),
            "difference": round(group1.mean() - group0.mean(), 2),
            "no_risk_median": round(group0.median(), 2),
            "risk_median": round(group1.median(), 2),
        })

    if numeric_rows:
        save_csv(pd.DataFrame(numeric_rows), "clinical_numeric_compare_table.csv")

    # 相關矩陣，讓 app_v5 畫熱力圖
    numeric_for_corr = heart_df.select_dtypes(include="number")
    corr_df = numeric_for_corr.corr().reset_index().rename(columns={"index": "feature"})
    save_csv(corr_df, "clinical_correlation_matrix.csv")


# =========================
# 5. 子專案三：模型分析
# =========================

def generate_model_insights(heart_df: pd.DataFrame, model_df: pd.DataFrame) -> None:
    """產生模型比較摘要、特徵重要度、模型資料故事。"""
    model_rows = []

    total_rows = len(heart_df) if not heart_df.empty else 0
    train_count = int(total_rows * 0.8) if total_rows else 0
    test_count = total_rows - train_count if total_rows else 0

    if not model_df.empty:
        # 取 AUC 最高作為最佳模型
        if "auc" in model_df.columns:
            best_row = model_df.sort_values("auc", ascending=False).iloc[0].to_dict()
        else:
            best_row = model_df.iloc[0].to_dict()

        model_rows.append({
            "metric": "dataset_total_rows",
            "value": total_rows,
            "description": "Cleaned_Heart_Data 總資料筆數",
        })
        model_rows.append({
            "metric": "train_rows_estimated",
            "value": train_count,
            "description": "模型訓練資料筆數，約 80%",
        })
        model_rows.append({
            "metric": "test_rows_estimated",
            "value": test_count,
            "description": "模型測試資料筆數，約 20%",
        })
        model_rows.append({
            "metric": "model_count",
            "value": len(model_df),
            "description": "比較的演算法數量，不是取 3 筆資料",
        })
        model_rows.append({
            "metric": "best_model",
            "value": best_row.get("model_name", "unknown"),
            "description": "依 AUC 選出的最佳模型",
        })
        model_rows.append({
            "metric": "best_auc",
            "value": round(float(best_row.get("auc", 0)), 4),
            "description": "最佳模型 AUC",
        })

        save_csv(pd.DataFrame(model_rows), "model_dashboard_summary.csv")
        save_csv(model_df, "model_comparison_for_dashboard.csv")

    # 特徵重要度
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)

            estimator = None

            if hasattr(model, "named_steps") and "model" in model.named_steps:
                estimator = model.named_steps["model"]
            else:
                estimator = model

            importance_values = None

            if hasattr(estimator, "feature_importances_"):
                importance_values = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                importance_values = abs(estimator.coef_[0])

            if importance_values is not None:
                importance_df = pd.DataFrame({
                    "feature": FEATURE_COLUMNS[:len(importance_values)],
                    "importance": importance_values,
                }).sort_values("importance", ascending=False)

                save_csv(importance_df, "model_feature_importance.csv")
            else:
                print("目前模型沒有 feature_importances_ 或 coef_，略過特徵重要度。")

        except Exception as error:
            print("讀取模型或產生特徵重要度失敗：")
            print(error)


# =========================
# 6. 子專案四：成本效益與資源配置 Demo
# =========================

def generate_cost_effectiveness_demo() -> None:
    """
    產生 MVP 模擬成本效益資料。
    注意：這不是正式醫療收費資料，只是課程專題展示用。
    """
    cost_df = pd.DataFrame([
        {
            "package_name": "基本健康追蹤",
            "included_items": "血壓、血糖、體重、運動、睡眠紀錄",
            "estimated_cost": 300,
            "expected_benefit": 0.45,
            "recommended_status": "穩定觀察",
            "business_note": "成本低，適合低風險或穩定追蹤族群",
        },
        {
            "package_name": "基礎檢查組合",
            "included_items": "血壓、血脂、血糖、生活型態評估",
            "estimated_cost": 800,
            "expected_benefit": 0.62,
            "recommended_status": "建議加強追蹤",
            "business_note": "適合中風險族群，兼顧成本與初步檢查完整度",
        },
        {
            "package_name": "心電圖追蹤組合",
            "included_items": "基礎檢查 + 靜息心電圖",
            "estimated_cost": 1500,
            "expected_benefit": 0.76,
            "recommended_status": "建議加強追蹤",
            "business_note": "適合有心電圖異常或症狀訊號者",
        },
        {
            "package_name": "完整心血管評估",
            "included_items": "基礎檢查 + 心電圖 + 運動測試 + 醫療評估",
            "estimated_cost": 3500,
            "expected_benefit": 0.88,
            "recommended_status": "建議優先關注",
            "business_note": "適合高風險或需進一步醫療評估者",
        },
    ])

    cost_df["benefit_per_1000_cost"] = (cost_df["expected_benefit"] / cost_df["estimated_cost"] * 1000).round(3)

    save_csv(cost_df, "cost_effectiveness_demo.csv")

    resource_df = pd.DataFrame([
        {"priority_order": 1, "status": "建議優先關注", "daily_slots": 5, "suggested_action": "優先安排完整心血管評估"},
        {"priority_order": 2, "status": "建議加強追蹤", "daily_slots": 10, "suggested_action": "安排基礎檢查或心電圖追蹤"},
        {"priority_order": 3, "status": "穩定觀察", "daily_slots": 20, "suggested_action": "健康追蹤與定期衛教"},
    ])

    save_csv(resource_df, "resource_allocation_demo.csv")


# =========================
# 7. 子專案五 / 六：使用者流程、健康管理與 KPI
# =========================

def generate_user_and_health_insights(user_df: pd.DataFrame, health_df: pd.DataFrame) -> None:
    """產生使用者預測摘要、健康追蹤異常摘要與 KPI。"""
    kpi_rows = []

    prediction_count = len(user_df) if not user_df.empty else 0
    health_log_count = len(health_df) if not health_df.empty else 0

    kpi_rows.append({"kpi": "prediction_count", "value": prediction_count, "description": "完成 AI 風險評估筆數"})
    kpi_rows.append({"kpi": "health_log_count", "value": health_log_count, "description": "健康追蹤紀錄筆數"})

    if not user_df.empty:
        if "risk_score" in user_df.columns:
            kpi_rows.append({
                "kpi": "average_risk_score",
                "value": round(user_df["risk_score"].mean(), 4),
                "description": "使用者預測紀錄平均 risk_score",
            })

        status_source_col = None
        for col in ["public_status", "internal_risk_level", "risk_level_zh", "risk_level"]:
            if col in user_df.columns:
                status_source_col = col
                break

        if status_source_col:
            status_df = user_df.copy()
            status_df["status_group"] = status_df[status_source_col].apply(status_zh)
            status_counts = status_df["status_group"].value_counts().reset_index()
            status_counts.columns = ["status_group", "count"]
            status_counts["percent"] = (status_counts["count"] / len(status_df) * 100).round(2)

            save_csv(status_counts, "user_status_distribution.csv")

            priority_count = int((status_df["status_group"] == "建議優先關注").sum())
            kpi_rows.append({
                "kpi": "priority_attention_count",
                "value": priority_count,
                "description": "需要優先關注的使用者筆數",
            })

    if not health_df.empty:
        alert_rows = []

        if "systolic_bp" in health_df.columns:
            bp_high_count = int((health_df["systolic_bp"] >= 140).sum())
            alert_rows.append({
                "alert_type": "systolic_bp_high",
                "alert_name": "收縮壓偏高",
                "count": bp_high_count,
                "suggestion": "建議規律量測血壓並記錄趨勢",
            })

        if "blood_sugar" in health_df.columns:
            sugar_high_count = int((health_df["blood_sugar"] >= 126).sum())
            alert_rows.append({
                "alert_type": "blood_sugar_high",
                "alert_name": "血糖偏高",
                "count": sugar_high_count,
                "suggestion": "建議追蹤血糖與代謝風險",
            })

        if "exercise_minutes" in health_df.columns:
            low_exercise_count = int((health_df["exercise_minutes"] < 20).sum())
            alert_rows.append({
                "alert_type": "exercise_low",
                "alert_name": "運動量不足",
                "count": low_exercise_count,
                "suggestion": "建議逐步增加每週活動量",
            })

        if "sleep_hours" in health_df.columns:
            low_sleep_count = int((health_df["sleep_hours"] < 6).sum())
            alert_rows.append({
                "alert_type": "sleep_low",
                "alert_name": "睡眠時數偏少",
                "count": low_sleep_count,
                "suggestion": "建議建立規律睡眠紀錄",
            })

        if alert_rows:
            save_csv(pd.DataFrame(alert_rows), "health_alert_summary.csv")

    # 六個子專案進度卡
    subproject_df = pd.DataFrame([
        {
            "subproject_id": 1,
            "subproject_name": "資料清理、資料庫設計與使用者輸入資料建置",
            "short_name": "資料庫",
            "color_group": "blue",
            "status": "已完成 MVP",
            "main_outputs": "cleaned_heart_data.csv、heart_disease_project.db、data_quality_report.csv",
        },
        {
            "subproject_id": 2,
            "subproject_name": "臨床指標關聯分析與健康建議依據",
            "short_name": "臨床分析",
            "color_group": "teal",
            "status": "展示資料已產生",
            "main_outputs": "clinical_categorical_risk_table.csv、clinical_numeric_compare_table.csv",
        },
        {
            "subproject_id": 3,
            "subproject_name": "AI 風險預測模型與使用者即時預測",
            "short_name": "AI模型",
            "color_group": "purple",
            "status": "已完成 MVP",
            "main_outputs": "heart_risk_model.pkl、model_comparison.csv、risk_score_report.csv",
        },
        {
            "subproject_id": 4,
            "subproject_name": "檢查成本效益與風險分級資源配置分析",
            "short_name": "成本效益",
            "color_group": "orange",
            "status": "MVP 模擬資料",
            "main_outputs": "cost_effectiveness_demo.csv、resource_allocation_demo.csv",
        },
        {
            "subproject_id": 5,
            "subproject_name": "使用者輸入、分流流程與健康管理儀表板應用",
            "short_name": "使用者流程",
            "color_group": "pink",
            "status": "已完成初版",
            "main_outputs": "Streamlit app_v5、user_prediction_result.csv",
        },
        {
            "subproject_id": 6,
            "subproject_name": "個人化健康管理建議與追蹤模組",
            "short_name": "健康追蹤",
            "color_group": "green",
            "status": "已完成初版",
            "main_outputs": "health_management_logs.csv、health_alert_summary.csv",
        },
    ])

    save_csv(subproject_df, "subproject_overview.csv")
    save_csv(pd.DataFrame(kpi_rows), "dashboard_kpi_summary.csv")


# =========================
# 8. 主程式
# =========================

def main() -> None:
    print("=" * 60)
    print("開始產生六子專案儀表板分析資料")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    heart_df = read_csv_safely(DATA_PATH)
    model_df = read_csv_safely(MODEL_COMPARISON_PATH)
    user_df = read_csv_safely(USER_PREDICTION_PATH)
    health_df = read_csv_safely(HEALTH_LOG_PATH)

    print(f"訓練資料筆數：{len(heart_df) if not heart_df.empty else 0}")
    print(f"模型比較筆數：{len(model_df) if not model_df.empty else 0}")
    print(f"使用者預測筆數：{len(user_df) if not user_df.empty else 0}")
    print(f"健康追蹤筆數：{len(health_df) if not health_df.empty else 0}")
    print()

    generate_data_quality_insights(heart_df)
    generate_clinical_feature_insights(heart_df)
    generate_model_insights(heart_df, model_df)
    generate_cost_effectiveness_demo()
    generate_user_and_health_insights(user_df, health_df)

    print()
    print("=" * 60)
    print("第 5-1 步完成！")
    print(f"所有儀表板分析資料已輸出到：{OUTPUT_DIR}")
    print("下一步可以建立 dashboard/app_v5.py，讀取這些資料做六子專案展示版儀表板。")
    print("=" * 60)


if __name__ == "__main__":
    main()
