"""
03_train_model.py

AI 智慧心血管疾病風險預測與健康管理系統
第一階段 MVP - 第 3 步：
使用 cleaned_heart_data.csv 訓練 AI 心血管風險預測模型。

執行位置：
    ~/Desktop/AI_HeartHealth_System

執行指令：
    python3 03_train_model.py

注意：
本模型僅作為課程專題、健康風險評估與決策支援原型，
不取代醫師診斷、醫療處置或正式臨床判斷。
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# =========================
# 1. 專案路徑設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_heart_data.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "heart_risk_model.pkl"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
RISK_SCORE_REPORT_PATH = REPORTS_DIR / "risk_score_report.csv"

TARGET_COLUMN = "target"

# 第一版 MVP 先使用 Cleveland 原始核心特徵
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# =========================
# 2. 工具函式
# =========================

def check_file_exists(file_path: Path) -> None:
    """確認資料檔案是否存在。"""
    if not file_path.exists():
        print("找不到資料檔案")
        print(f"目前程式尋找的位置：{file_path}")
        sys.exit(1)


def load_data() -> pd.DataFrame:
    """讀取清理後資料。"""
    df = pd.read_csv(DATA_PATH)
    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """確認模型需要的欄位都有存在。"""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = []
    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        print("資料缺少以下欄位，無法訓練模型：")
        for column in missing_columns:
            print(f"- {column}")
        sys.exit(1)


def build_models() -> dict:
    """建立要比較的模型。"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    return models


def build_pipeline(model):
    """
    建立模型流程：
    1. 缺失值用中位數補值
    2. 數值標準化
    3. 放入分類模型
    """
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    return pipeline


def evaluate_model(model_name, pipeline, x_train, x_test, y_train, y_test) -> dict:
    """訓練並評估單一模型。"""
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    result = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }

    return result


def risk_level_from_score(score: float) -> str:
    """依照風險分數轉換為低、中、高風險。"""
    if score < 0.30:
        return "low_risk"
    elif score < 0.70:
        return "medium_risk"
    else:
        return "high_risk"


# =========================
# 3. 主程式
# =========================

def main() -> None:
    print("開始執行：AI 心血管風險預測模型訓練")
    print(f"資料來源：{DATA_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    check_file_exists(DATA_PATH)

    df = load_data()
    check_required_columns(df)

    print()
    print("=" * 50)
    print("資料基本資訊")
    print("=" * 50)
    print(f"資料筆數：{df.shape[0]}")
    print(f"欄位數：{df.shape[1]}")
    print()
    print("target 分布：")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    # 取出模型特徵 X 與目標 y
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 切分訓練集與測試集
    # stratify=y 可以讓訓練集與測試集的 target 比例較一致
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print()
    print("=" * 50)
    print("資料切分結果")
    print("=" * 50)
    print(f"訓練資料筆數：{x_train.shape[0]}")
    print(f"測試資料筆數：{x_test.shape[0]}")

    models = build_models()
    comparison_results = []
    trained_pipelines = {}

    print()
    print("=" * 50)
    print("開始訓練與比較模型")
    print("=" * 50)

    for model_name, model in models.items():
        pipeline = build_pipeline(model)
        result = evaluate_model(model_name, pipeline, x_train, x_test, y_train, y_test)

        comparison_results.append(result)
        trained_pipelines[model_name] = pipeline

        print()
        print(f"模型：{model_name}")
        print(f"Accuracy：{result['accuracy']:.4f}")
        print(f"Precision：{result['precision']:.4f}")
        print(f"Recall：{result['recall']:.4f}")
        print(f"F1：{result['f1']:.4f}")
        print(f"AUC：{result['auc']:.4f}")

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values(by="auc", ascending=False)

    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False, encoding="utf-8-sig")

    best_model_name = comparison_df.iloc[0]["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    print()
    print("=" * 50)
    print("最佳模型")
    print("=" * 50)
    print(f"最佳模型：{best_model_name}")

    # 將最佳模型重新用全部資料訓練，作為 MVP 正式使用模型
    final_model = build_pipeline(models[best_model_name])
    final_model.fit(x, y)

    joblib.dump(final_model, MODEL_PATH)

    # 儲存模型欄位設定，之後 Streamlit 表單與即時預測會用到
    feature_schema = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "risk_level_rule": {
            "low_risk": "risk_score < 0.30",
            "medium_risk": "0.30 <= risk_score < 0.70",
            "high_risk": "risk_score >= 0.70"
        },
        "disclaimer": "本模型僅作為健康風險評估與決策支援原型，不取代醫師診斷。"
    }

    FEATURE_SCHEMA_PATH.write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

    # 產生全資料風險分數報表，方便後續做儀表板
    risk_scores = final_model.predict_proba(x)[:, 1]

    risk_report = df.copy()
    risk_report["risk_score"] = risk_scores
    risk_report["risk_level"] = risk_report["risk_score"].apply(risk_level_from_score)

    risk_report.to_csv(RISK_SCORE_REPORT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=" * 50)
    print("輸出檔案")
    print("=" * 50)
    print(f"模型檔案：{MODEL_PATH}")
    print(f"模型欄位設定：{FEATURE_SCHEMA_PATH}")
    print(f"模型比較報表：{MODEL_COMPARISON_PATH}")
    print(f"風險分數報表：{RISK_SCORE_REPORT_PATH}")

    print()
    print("第 3 步完成！")
    print("下一步可以做：04_user_prediction.py，建立使用者即時輸入預測功能。")


if __name__ == "__main__":
    main()
