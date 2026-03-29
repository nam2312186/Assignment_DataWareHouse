"""
=============================================================
Pipeline Orchestrator — Chạy E → T → L theo thứ tự
=============================================================
"""

import time
from .extract   import extract_all
from .transform import transform_all
from .load      import load_all, verify_dw


def run_pipeline() -> str:
    """
    Chạy toàn bộ ETL pipeline:
      Stage 1 — Extract   : CSV → Parquet (staging)
      Stage 2 — Transform : Business logic transformations
      Stage 3 — Load      : DataFrames + DuckDB SQL → DuckDB DW

    Returns: đường dẫn đến file DW .duckdb
    """
    wall_start = time.time()

    # ── Stage 1: Extract ────────────────────────────────────
    t0 = time.time()
    extract_all()
    t_extract = time.time() - t0

    # ── Stage 2: Transform ──────────────────────────────────
    t0 = time.time()
    transformed = transform_all()
    t_transform = time.time() - t0

    # ── Stage 3: Load ───────────────────────────────────────
    t0 = time.time()
    db_path = load_all(transformed)
    t_load = time.time() - t0

    # ── Verify ──────────────────────────────────────────────
    verify_dw()

    # ── Summary ─────────────────────────────────────────────
    total = time.time() - wall_start
    print("=" * 60)
    print("  ETL PIPELINE — KẾT QUẢ")
    print("=" * 60)
    print(f"  Stage 1 Extract  : {t_extract:>7.1f}s")
    print(f"  Stage 2 Transform: {t_transform:>7.1f}s")
    print(f"  Stage 3 Load     : {t_load:>7.1f}s")
    print(f"  ─────────────────────────────")
    print(f"  Tổng cộng        : {total:>7.1f}s  ({total/60:.1f} phút)")
    print(f"\n  ✅ Data Warehouse: {db_path}")
    print("=" * 60 + "\n")

    return db_path
