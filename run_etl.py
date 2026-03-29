"""
=============================================================
Entry Point — Chạy toàn bộ pipeline ETL + build Data Marts

Usage:
    .venv\\Scripts\\python run_etl.py

Pipeline:
    Stage 1 EXTRACT   : data/raws/*.csv      → data/staging/*.parquet
    Stage 2 TRANSFORM : staging/*.parquet    → cleaned DataFrames
    Stage 3 LOAD      : DataFrames + SQL     → data/warehouse/instacart_dw.duckdb
    MART 1            : DW query             → data/warehouse/marts/mart_basket.csv
    MART 2            : DW query             → data/warehouse/marts/mart_user_item.csv
=============================================================
"""

import sys
import time

# Ensure src/ is on path khi chạy từ root project
sys.path.insert(0, ".")

from src.etl.pipeline       import run_pipeline
from src.marts.mart_basket  import build_mart_basket
from src.marts.mart_user_item import build_mart_user_item


def main():
    print("\n" + "█" * 60)
    print("  INSTACART DATA WAREHOUSE — FULL ETL PIPELINE")
    print("█" * 60)

    total_start = time.time()

    # ── ETL Pipeline ────────────────────────────────────────
    db_path = run_pipeline()

    # ── Build Data Marts ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BUILDING DATA MARTS")
    print("=" * 60)

    build_mart_basket()
    build_mart_user_item()

    # ── Done ────────────────────────────────────────────────
    total = time.time() - total_start
    print("\n" + "█" * 60)
    print(f"  ✅ TẤT CẢ HOÀN TẤT trong {total:.1f}s ({total/60:.1f} phút)")
    print()
    print("  Files đã tạo:")
    print(f"    DW      → data/warehouse/instacart_dw.duckdb")
    print(f"    Mart 1  → data/warehouse/marts/mart_basket.csv")
    print(f"    Mart 2  → data/warehouse/marts/mart_user_item.csv")
    print()
    print("  Bước tiếp theo:")
    print("    → Mở notebook/apriori.ipynb để train Apriori/FP-Growth")
    print("    → Mở notebook/apriori_vs_gnn_comparison.ipynb để train LightGCN")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline bị dừng bởi người dùng.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
