"""
01_import_excel_to_database.py

AI 智慧心血管疾病風險預測與健康管理系統
第一階段 MVP - 第 1 步：
將 Excel 資料集匯入成 CSV 與 SQLite 資料庫。

執行位置：
    ~/Desktop/AI_HeartHealth_System

執行指令：
    python3 01_import_excel_to_database.py
"""

from pathlib import Path
import sqlite3
import sys

import pandas as pd


# =========================
# 1. 專案路徑設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

RAW_EXCEL_PATH = BASE_DIR / "data" / "raw" / "AI心血管專題_資料集包_cleaned_dataset.xlsx"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATABASE_DIR = BASE_DIR / "database"

CSV_OUTPUT_PATH = PROCESSED_DIR / "cleaned_heart_data.csv"
DB_OUTPUT_PATH = DATABASE_DIR / "heart_disease_project.db"

TABLE_NAME = "cleaned_patient_data"


# =========================
# 2. 工具函式
# =========================

def check_file_exists(file_path: Path) -> None:
    """確認 Excel 檔案是否存在。"""
    if not file_path.exists():
        print("找不到 Excel 資料集")
        print(f"目前程式尋找的位置：{file_path}")
        print()
        print("請確認檔案是否放在：")
        print("~/Desktop/AI_HeartHealth_System/data/raw/AI心血管專題_資料集包_cleaned_dataset.xlsx")
        sys.exit(1)


def create_folders() -> None:
    """建立輸出資料夾。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)


def load_excel(file_path: Path) -> pd.DataFrame:
    """讀取 Excel 檔案，優先讀取 Cleaned_Heart_Data 工作表。"""
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        print("偵測到 Excel 工作表：")
        for index, sheet_name in enumerate(sheet_names, start=1):
            print(f"{index}. {sheet_name}")

        target_sheet_name = "Cleaned_Heart_Data"

        if target_sheet_name in sheet_names:
            selected_sheet = target_sheet_name
        else:
            print()
            print("找不到 Cleaned_Heart_Data，程式會先改讀第一個工作表。")
            print("如果資料筆數不對，請把上面的工作表名稱截圖給我。")
            selected_sheet = sheet_names[0]

        print()
        print(f"目前讀取工作表：{selected_sheet}")

        df = pd.read_excel(file_path, sheet_name=selected_sheet)
        return df

    except ImportError:
        print("讀取 Excel 失敗：缺少 openpyxl 套件")
        print("請先執行：")
        print("python3 -m pip install openpyxl")
        sys.exit(1)

    except Exception as error:
        print("讀取 Excel 時發生錯誤")
        print(error)
        sys.exit(1)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理欄位名稱：
    1. 去除前後空白
    2. 空白改成底線
    3. 避免欄位名稱重複
    """
    new_columns = []
    used_columns = set()

    for column in df.columns:
        new_column = str(column).strip()
        new_column = new_column.replace(" ", "_")
        new_column = new_column.replace("\n", "_")
        new_column = new_column.replace("\t", "_")

        if new_column == "" or new_column.lower() == "nan":
            new_column = "unnamed_column"

        original_column = new_column
        counter = 1

        while new_column in used_columns:
            counter += 1
            new_column = f"{original_column}_{counter}"

        used_columns.add(new_column)
        new_columns.append(new_column)

    df.columns = new_columns
    return df


def show_data_summary(df: pd.DataFrame) -> None:
    """顯示資料基本資訊。"""
    print()
    print("=" * 50)
    print("資料基本資訊")
    print("=" * 50)
    print(f"資料筆數：{df.shape[0]}")
    print(f"欄位數：{df.shape[1]}")

    print()
    print("欄位名稱：")
    for column in df.columns:
        print(f"- {column}")

    print()
    print("缺失值統計：")
    print(df.isna().sum())

    print()
    print("前 5 筆資料：")
    print(df.head())


def export_to_csv(df: pd.DataFrame) -> None:
    """匯出 CSV。"""
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print()
    print(f"CSV 已匯出：{CSV_OUTPUT_PATH}")


def export_to_sqlite(df: pd.DataFrame) -> None:
    """匯出 SQLite 資料庫。"""
    try:
        with sqlite3.connect(DB_OUTPUT_PATH) as conn:
            df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            row_count = cursor.fetchone()[0]

        print(f"SQLite 資料庫已建立：{DB_OUTPUT_PATH}")
        print(f"資料表名稱：{TABLE_NAME}")
        print(f"SQLite 資料筆數：{row_count}")

    except Exception as error:
        print("匯出 SQLite 時發生錯誤")
        print(error)
        sys.exit(1)


# =========================
# 3. 主程式
# =========================

def main() -> None:
    print("開始執行：Excel 匯入 CSV 與 SQLite")
    print(f"專案位置：{BASE_DIR}")
    print(f"Excel 位置：{RAW_EXCEL_PATH}")

    check_file_exists(RAW_EXCEL_PATH)
    create_folders()

    df = load_excel(RAW_EXCEL_PATH)
    df = clean_column_names(df)

    show_data_summary(df)

    export_to_csv(df)
    export_to_sqlite(df)

    print()
    print("第 1 步完成！")
    print("你現在已經有：")
    print(f"1. CSV 檔案：{CSV_OUTPUT_PATH}")
    print(f"2. SQLite 資料庫：{DB_OUTPUT_PATH}")
    print()
    print("下一步可以做：02_check_database.py，確認資料庫內容與欄位品質。")


if __name__ == "__main__":
    main()
