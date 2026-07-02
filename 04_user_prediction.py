"""
04_user_prediction.py

AI 智慧心血管疾病風險預測與健康管理系統
第一階段 MVP - 第 4 步：
建立使用者即時輸入預測功能。

功能：
1. 載入 models/heart_risk_model.pkl
2. 讓使用者在 Terminal 輸入 Cleveland 核心特徵
3. 即時輸出 risk_score、risk_level 與健康建議
4. 將使用者輸入與預測結果寫入 SQLite
5. 將預測結果追加保存到 reports/user_prediction_result.csv

執行位置：
    ~/Desktop/AI_HeartHealth_System

執行指令：
    python3 04_user_prediction.py

注意：
本系統僅作為課程專題、健康風險評估與決策支援原型，
不取代醫師診斷、醫療處置或正式臨床判斷。
"""

from pathlib import Path
from datetime import datetime
import json
import sqlite3
import sys

import joblib
import pandas as pd


# =========================
# 1. 專案路徑設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "heart_risk_model.pkl"
FEATURE_SCHEMA_PATH = BASE_DIR / "models" / "feature_schema.json"
DB_PATH = BASE_DIR / "database" / "heart_disease_project.db"
REPORTS_DIR = BASE_DIR / "reports"
USER_RESULT_CSV_PATH = REPORTS_DIR / "user_prediction_result.csv"

USER_INPUT_TABLE = "user_input_records"
PREDICTION_TABLE = "model_prediction_result"


# =========================
# 2. 欄位說明
# =========================

FEATURE_HELP = {
    "age": "年齡，例如 45、60",
    "sex": "性別：1=男性，0=女性",
    "cp": "胸痛型態：1=典型心絞痛，2=非典型心絞痛，3=非心絞痛疼痛，4=無症狀",
    "trestbps": "靜息血壓 mmHg，例如 120、140",
    "chol": "膽固醇 mg/dl，例如 180、240",
    "fbs": "空腹血糖是否大於 120 mg/dl：1=是，0=否",
    "restecg": "靜息心電圖：0=正常，1=ST-T異常，2=左心室肥大可能",
    "thalach": "最大心率，例如 150",
    "exang": "運動誘發心絞痛：1=是，0=否",
    "oldpeak": "運動後 ST depression，例如 0、1.5、2.3",
    "slope": "ST 斜率：1=上升，2=平坦，3=下降",
    "ca": "螢光顯影血管數：0~3",
    "thal": "Thal 檢查：3=正常，6=固定缺陷，7=可逆缺陷",
}

ALLOWED_VALUES = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}


# =========================
# 3. 工具函式
# =========================

def check_file_exists(file_path: Path, label: str) -> None:
    """確認必要檔案是否存在。"""
    if not file_path.exists():
        print(f"找不到 {label}")
        print(f"目前程式尋找的位置：{file_path}")
        print()
        print("請確認你已經完成 03_train_model.py，並成功產生模型檔。")
        sys.exit(1)


def load_feature_schema() -> dict:
    """讀取模型欄位設定。"""
    with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as file:
        schema = json.load(file)
    return schema


def load_model():
    """載入訓練好的模型。"""
    model = joblib.load(MODEL_PATH)
    return model


def get_user_value(column_name: str) -> float:
    """讓使用者輸入單一欄位，並做基本驗證。"""
    while True:
        print()
        print(f"欄位：{column_name}")
        print(f"說明：{FEATURE_HELP.get(column_name, '請輸入數值')}")

        user_input = input("請輸入數值：").strip()

        try:
            value = float(user_input)

            # 若是有限制選項的欄位，檢查是否在允許值內
            if column_name in ALLOWED_VALUES:
                int_value = int(value)

                if value != int_value:
                    print("這個欄位需要輸入整數選項，請重新輸入。")
                    continue

                if int_value not in ALLOWED_VALUES[column_name]:
                    print(f"輸入值不在允許範圍內，允許值為：{ALLOWED_VALUES[column_name]}")
                    continue

                return int_value

            # 其他欄位做簡單合理範圍提醒
            if column_name == "age" and not (1 <= value <= 120):
                print("年齡看起來不太合理，請重新輸入。")
                continue

            if column_name in ["trestbps", "chol", "thalach"] and value <= 0:
                print("這個欄位應該大於 0，請重新輸入。")
                continue

            if column_name == "oldpeak" and value < 0:
                print("oldpeak 通常不會是負數，請重新輸入。")
                continue

            return value

        except ValueError:
            print("輸入格式錯誤，請輸入數字。")


def risk_level_from_score(score: float) -> str:
    """依照風險分數轉換低、中、高風險。"""
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


def build_suggestions(input_data: dict, risk_score: float, risk_level: str) -> list:
    """依風險等級與使用者輸入數值產生健康建議。"""
    suggestions = []

    if risk_level == "low_risk":
        suggestions.append("目前模型評估為低風險，建議維持規律作息、均衡飲食與定期健康追蹤。")
    elif risk_level == "medium_risk":
        suggestions.append("目前模型評估為中風險，建議持續追蹤血壓、血脂與血糖，並視情況安排健康檢查。")
    else:
        suggestions.append("目前模型評估為高風險，建議優先諮詢醫療人員，進一步評估心血管相關風險。")

    # 依照個別指標補充提醒
    if input_data.get("trestbps", 0) >= 140:
        suggestions.append("靜息血壓偏高，建議規律量測血壓並記錄趨勢。")

    if input_data.get("chol", 0) >= 240:
        suggestions.append("膽固醇數值偏高，建議追蹤血脂並留意飲食與回診建議。")

    if input_data.get("fbs", 0) == 1:
        suggestions.append("空腹血糖指標異常，建議追蹤血糖與代謝相關風險。")

    if input_data.get("exang", 0) == 1:
        suggestions.append("有運動誘發心絞痛訊號，建議避免忽視症狀，必要時尋求醫療評估。")

    if input_data.get("oldpeak", 0) >= 2:
        suggestions.append("oldpeak 數值較高，建議搭配心電圖或醫療人員判讀進一步評估。")

    suggestions.append("提醒：本結果僅供健康風險評估與課程專題展示，不代表正式診斷。")

    return suggestions


def create_database_tables() -> None:
    """建立使用者輸入與預測結果資料表。"""
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

        conn.commit()


def save_prediction_to_database(record_id: str, created_at: str, input_data: dict,
                                risk_score: float, risk_level: str, suggestions: list) -> None:
    """將使用者輸入與預測結果寫入 SQLite。"""
    prediction_id = "PRED_" + record_id
    suggestion_text = "；".join(suggestions)

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


def append_prediction_to_csv(record_id: str, created_at: str, input_data: dict,
                             risk_score: float, risk_level: str, suggestions: list) -> None:
    """將單次預測結果追加到 CSV。"""
    result = {
        "record_id": record_id,
        "created_at": created_at,
        **input_data,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_level_zh": risk_level_zh(risk_level),
        "suggestion_text": "；".join(suggestions),
    }

    result_df = pd.DataFrame([result])

    if USER_RESULT_CSV_PATH.exists():
        old_df = pd.read_csv(USER_RESULT_CSV_PATH)
        new_df = pd.concat([old_df, result_df], ignore_index=True)
    else:
        new_df = result_df

    new_df.to_csv(USER_RESULT_CSV_PATH, index=False, encoding="utf-8-sig")


# =========================
# 4. 主程式
# =========================

def main() -> None:
    print("開始執行：使用者即時風險預測")
    print("本功能會載入已訓練模型，並請你輸入一位使用者的資料。")
    print()
    print("提醒：本系統僅作為健康風險評估與決策支援原型，不取代醫師診斷。")

    check_file_exists(MODEL_PATH, "模型檔 heart_risk_model.pkl")
    check_file_exists(FEATURE_SCHEMA_PATH, "模型欄位設定 feature_schema.json")
    check_file_exists(DB_PATH, "SQLite 資料庫 heart_disease_project.db")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    schema = load_feature_schema()
    feature_columns = schema["feature_columns"]
    model = load_model()

    print()
    print("=" * 50)
    print("請依序輸入使用者資料")
    print("=" * 50)

    input_data = {}
    for column in feature_columns:
        input_data[column] = get_user_value(column)

    input_df = pd.DataFrame([input_data], columns=feature_columns)

    risk_score = float(model.predict_proba(input_df)[:, 1][0])
    risk_level = risk_level_from_score(risk_score)
    suggestions = build_suggestions(input_data, risk_score, risk_level)

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_id = "USER_" + datetime.now().strftime("%Y%m%d%H%M%S")

    create_database_tables()
    save_prediction_to_database(record_id, created_at, input_data, risk_score, risk_level, suggestions)
    append_prediction_to_csv(record_id, created_at, input_data, risk_score, risk_level, suggestions)

    print()
    print("=" * 50)
    print("預測結果")
    print("=" * 50)
    print(f"紀錄編號：{record_id}")
    print(f"risk_score：{risk_score:.4f}")
    print(f"risk_level：{risk_level_zh(risk_level)}")

    print()
    print("健康建議：")
    for index, suggestion in enumerate(suggestions, start=1):
        print(f"{index}. {suggestion}")

    print()
    print("=" * 50)
    print("輸出完成")
    print("=" * 50)
    print(f"預測結果 CSV：{USER_RESULT_CSV_PATH}")
    print(f"SQLite 資料表：{USER_INPUT_TABLE}, {PREDICTION_TABLE}")
    print()
    print("第 4 步完成！下一步可以建立簡易健康建議模組或 Streamlit 儀表板。")


if __name__ == "__main__":
    main()
