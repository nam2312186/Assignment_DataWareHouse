"""
=============================================================
Stage 1 — EXTRACT
=============================================================
Mục tiêu:
  - Đọc 6 file CSV thô từ data/raws/
  - Ép kiểu dữ liệu (dtype casting) để tiết kiệm RAM
  - Lưu ra data/staging/ dưới dạng Parquet (nhanh hơn CSV khi đọc lại)

Nguyên tắc: KHÔNG làm business logic ở bước này.
            Chỉ đọc → ép kiểu → lưu.
=============================================================
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR     = Path("data/raws")
STAGING_DIR = Path("data/staging")

# ── Khai báo dtype cho từng file để pandas đọc đúng kiểu, tiết kiệm RAM ──────
SCHEMA = {
    "aisles.csv": {
        "aisle_id": "int16",
        "aisle":    "str",
    },
    "departments.csv": {
        "department_id": "int8",
        "department":    "str",
    },
    "products.csv": {
        "product_id":    "int32",
        "product_name":  "str",
        "aisle_id":      "int16",
        "department_id": "int8",
    },
    "orders.csv": {
        "order_id":               "int32",
        "user_id":                "int32",
        "eval_set":               "str",
        "order_number":           "int16",
        "order_dow":              "int8",
        "order_hour_of_day":      "int8",
        "days_since_prior_order": "float32",   # có null → dùng float
    },
    "order_products__train.csv": {
        "order_id":           "int32",
        "product_id":         "int32",
        "add_to_cart_order":  "int16",
        "reordered":          "int8",
    },
    "order_products__prior.csv": {
        "order_id":           "int32",
        "product_id":         "int32",
        "add_to_cart_order":  "int16",
        "reordered":          "int8",
    },
}

CHUNK_SIZE = 1_500_000   # Số dòng mỗi chunk khi đọc file 32M dòng


def _extract_file(filename: str) -> pd.DataFrame:
    """Đọc một file CSV → DataFrame, xử lý chunk cho file lớn."""
    src = RAW_DIR / filename
    if not src.exists():
        raise FileNotFoundError(f"[EXTRACT] Không tìm thấy: {src}")

    size_mb = src.stat().st_size / 1024 / 1024
    print(f"\n  → {filename}  ({size_mb:.1f} MB)")

    dtype = SCHEMA[filename]

    # File 32M dòng → đọc theo chunk để tránh tràn RAM
    if filename == "order_products__prior.csv":
        chunks = []
        reader = pd.read_csv(src, dtype=dtype, chunksize=CHUNK_SIZE)
        total_rows = 0
        with tqdm(desc="    chunks", unit="chunk", ncols=70) as pbar:
            for chunk in reader:
                chunks.append(chunk)
                total_rows += len(chunk)
                pbar.update(1)
                pbar.set_postfix(rows=f"{total_rows:,}")
        df = pd.concat(chunks, ignore_index=True)
        del chunks
    else:
        df = pd.read_csv(src, dtype=dtype)

    print(f"    ✓ {len(df):,} dòng  |  {df.shape[1]} cột  |  dtype OK")
    return df


def extract_all() -> None:
    """Chạy Extract cho tất cả 6 file. Lưu staging Parquet."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  STAGE 1 — EXTRACT")
    print("=" * 60)
    print(f"  Nguồn  : {RAW_DIR}/")
    print(f"  Đích   : {STAGING_DIR}/")

    for filename in SCHEMA:
        df  = _extract_file(filename)
        out = STAGING_DIR / filename.replace(".csv", ".parquet")
        df.to_parquet(out, index=False, engine="pyarrow")
        print(f"    saved → {out.name}")

    print("\n  [EXTRACT] Hoàn tất!\n")
