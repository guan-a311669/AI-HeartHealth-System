"""
02_check_database.py

AI 智慧心血管疾病風險預測與健康管理系統
第一階段 MVP - 第 2 步：
檢查 CSV 與 SQLite 資料庫內容是否正確。

執行位置：
    ~/Desktop/AI_HeartHealth_System

執行指令：
    python3 02_check_database.py
"""

from pathlib import Path
import sqlite3
import sys

import pandas as pd


# =========================
# 1. 專案路徑設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_heart_data.csv"
DB_PATH = BASE_DIR / "database" / "heart_disease_project.db"
REPORTS_DIR = BASE_DIR / "reports"

TABLE_NAME = "cleaned_patient_data"

QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.csv"
SUMMARY_REPORT_PATH = REPORTS_DIR / "database_check_summary.txt"


# =========================
# 2. 工具函式
# =========================

def check_file_exists(file_path: Path, file_label: str) -> None:
    """確認指定檔案是否存在。"""
    if not file_path.exists():
        print(f"找不到 {file_label}")
        print(f"目前程式尋找的位置：{file_path}")
        sys.exit(1)


def read_csv_data() -> pd.DataFrame:
    """讀取 CSV 資料。"""
    df = pd.read_csv(CSV_PATH)
    return df


def read_sqlite_data() -> pd.DataFrame:
    """讀取 SQLite 資料表。"""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    return df


def show_sqlite_tables() -> list:
    """列出 SQLite 裡面的所有資料表。"""
    with sqlite3.connect(DB_PATH) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            conn
        )

    table_list = tables["name"].tolist()
    return table_list


def create_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """建立資料品質檢查表。"""
    report = pd.DataFrame({
        "column_name": df.columns,
        "dtype": [str(df[column].dtype) for column in df.columns],
        "missing_count": [df[column].isna().sum() for column in df.columns],
        "missing_rate": [df[column].isna().mean() for column in df.columns],
        "unique_count": [df[column].nunique() for column in df.columns],
    })

    return report


def check_target_column(df: pd.DataFrame) -> str:
    """檢查 target 欄位是否存在與分布。"""
    if "target" not in df.columns:
        return "找不到 target 欄位，請確認資料集是否已完成 target 建立。"

    target_counts = df["target"].value_counts(dropna=False).sort_index()
    result_text = "target 欄位分布：\n"

    for target_value, count in target_counts.items():
        result_text += f"target = {target_value}：{count} 筆\n"

    return result_text


def main() -> None:
    print("開始執行：資料庫與資料品質檢查")
    print(f"專案位置：{BASE_DIR}")
    print(f"CSV 位置：{CSV_PATH}")
    print(f"SQLite 位置：{DB_PATH}")
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    check_file_exists(CSV_PATH, "CSV 檔案 cleaned_heart_data.csv")
    check_file_exists(DB_PATH, "SQLite 資料庫 heart_disease_project.db")

    csv_df = read_csv_data()
    sqlite_df = read_sqlite_data()
    table_list = show_sqlite_tables()

    print("=" * 50)
    print("1. SQLite 資料表檢查")
    print("=" * 50)
    print("資料庫內的資料表：")
    for table in table_list:
        print(f"- {table}")

    print()
    print("=" * 50)
    print("2. CSV 與 SQLite 筆數檢查")
    print("=" * 50)
    print(f"CSV 資料筆數：{csv_df.shape[0]}")
    print(f"SQLite 資料筆數：{sqlite_df.shape[0]}")

    if csv_df.shape[0] == sqlite_df.shape[0]:
        print("筆數一致")
    else:
        print("筆數不一致，請回頭檢查 01_import_excel_to_database.py")

    print()
    print("=" * 50)
    print("3. 欄位檢查")
    print("=" * 50)
    print(f"CSV 欄位數：{csv_df.shape[1]}")
    print(f"SQLite 欄位數：{sqlite_df.shape[1]}")

    print()
    print("欄位名稱：")
    for column in csv_df.columns:
        print(f"- {column}")

    print()
    print("=" * 50)
    print("4. 缺失值檢查")
    print("=" * 50)
    missing_values = csv_df.isna().sum()
    print(missing_values)

    print()
    print("=" * 50)
    print("5. 資料型態檢查")
    print("=" * 50)
    print(csv_df.dtypes)

    print()
    print("=" * 50)
    print("6. target 欄位檢查")
    print("=" * 50)
    target_result = check_target_column(csv_df)
    print(target_result)

    print()
    print("=" * 50)
    print("7. 前 5 筆資料")
    print("=" * 50)
    print(csv_df.head())

    quality_report = create_quality_report(csv_df)
    quality_report.to_csv(QUALITY_REPORT_PATH, index=False, encoding="utf-8-sig")

    summary_text = f"""
AI 智慧心血管疾病風險預測與健康管理系統
資料庫檢查摘要

CSV 檔案：
{CSV_PATH}

SQLite 資料庫：
{DB_PATH}

SQLite 資料表：
{TABLE_NAME}

CSV 資料筆數：
{csv_df.shape[0]}

SQLite 資料筆數：
{sqlite_df.shape[0]}

欄位數：
{csv_df.shape[1]}

target 檢查：
{target_result}

資料品質報表：
{QUALITY_REPORT_PATH}
"""

    SUMMARY_REPORT_PATH.write_text(summary_text, encoding="utf-8")

    print()
    print("第 2 步完成！")
    print(f"資料品質報表已輸出：{QUALITY_REPORT_PATH}")
    print(f"資料庫檢查摘要已輸出：{SUMMARY_REPORT_PATH}")
    print()
    print("如果資料筆數是 303，且 SQLite 筆數也是 303，就代表資料匯入成功。")


if __name__ == "__main__":
    main()
