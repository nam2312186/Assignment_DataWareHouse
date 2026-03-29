"""
=============================================================
Stage 2 — TRANSFORM  (⭐ Bước bắt buộc)
=============================================================
Thực hiện tất cả business logic biến đổi dữ liệu:

  dim_department  → rename cột + audit timestamp
  dim_aisle       → rename cột + audit timestamp
  dim_product     → join aisles/depts + tính reorder_rate, order_frequency
  dim_user        → tổng hợp hành vi mua sắm per user từ orders
  dim_order       → fill null, map tên thứ, phân loại time_of_day
  fact_order_items→ (xử lý bởi load.py dùng DuckDB SQL trực tiếp từ Parquet
                      vì 32M+ dòng — quá lớn để pandas xử lý trong RAM)

Input  : data/staging/*.parquet
Output : dict[table_name → DataFrame]  (chỉ dimension tables)
         Fact table được transform + load trong load.py bằng DuckDB SQL
=============================================================
"""

import pandas as pd
import duckdb
from datetime import datetime
from pathlib import Path

STAGING_DIR = Path("data/staging")


def _now() -> str:
    """Audit timestamp — ghi lại thời điểm ETL."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parquet(name: str) -> str:
    """Trả về đường dẫn staging parquet (dưới dạng string cho DuckDB)."""
    return str(STAGING_DIR / f"{name}.parquet")


def _load(name: str) -> pd.DataFrame:
    """Đọc staging parquet vào pandas DataFrame."""
    path = STAGING_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"[TRANSFORM] Không tìm thấy staging file: {path}\n"
            "→ Hãy chạy Extract trước: run_etl.py hoặc extract_all()"
        )
    return pd.read_parquet(path)


# ═══════════════════════════════════════════════════════════
#  DIMENSION TRANSFORMS
# ═══════════════════════════════════════════════════════════

def transform_dim_department() -> pd.DataFrame:
    """
    Transforms:
      ① Rename 'department' → 'department_name'  (chuẩn hoá tên cột)
      ② Thêm 'created_at'  (audit column — biết dữ liệu nạp lúc nào)
    """
    print("    → dim_department ...")
    df = _load("departments")
    df = df.rename(columns={"department": "department_name"})
    df["created_at"] = _now()
    return df[["department_id", "department_name", "created_at"]]


def transform_dim_aisle() -> pd.DataFrame:
    """
    Transforms:
      ① Rename 'aisle' → 'aisle_name'  (chuẩn hoá tên cột)
      ② Thêm 'created_at'
    """
    print("    → dim_aisle ...")
    df = _load("aisles")
    df = df.rename(columns={"aisle": "aisle_name"})
    df["created_at"] = _now()
    return df[["aisle_id", "aisle_name", "created_at"]]


def transform_dim_product() -> pd.DataFrame:
    """
    Transforms:
      ① Join products ← aisles  (lấy aisle_name)
      ② Join products ← departments  (lấy department_name)
      ③ Tính reorder_rate per product từ 32M dòng prior
         → Dùng DuckDB in-memory query trên Parquet (nhanh, ít RAM)
      ④ Tính order_frequency  (tổng số lần xuất hiện trong prior)
      ⑤ Fill null reorder_rate/order_frequency = 0 (sản phẩm chưa mua bao giờ)
      ⑥ Thêm 'created_at'
    """
    print("    → dim_product  (join aisles + departments + tính reorder stats từ 32M rows)...")

    products = _load("products")
    aisles   = _load("aisles").rename(columns={"aisle": "aisle_name"})
    depts    = _load("departments").rename(columns={"department": "department_name"})

    # ③④ Dùng DuckDB để tính stats từ file prior 32M dòng mà không load vào RAM
    prior_path = _parquet("order_products__prior")
    con = duckdb.connect()
    product_stats = con.execute(f"""
        SELECT
            product_id,
            COUNT(*)        AS order_frequency,
            AVG(reordered)  AS reorder_rate
        FROM read_parquet('{prior_path}')
        GROUP BY product_id
    """).df()
    con.close()

    # ①② Join
    df = (
        products
        .merge(aisles[["aisle_id", "aisle_name"]], on="aisle_id", how="left")
        .merge(depts[["department_id", "department_name"]], on="department_id", how="left")
        .merge(product_stats, on="product_id", how="left")
    )

    # ⑤ Fill null
    df["reorder_rate"]    = df["reorder_rate"].fillna(0).round(4)
    df["order_frequency"] = df["order_frequency"].fillna(0).astype("int32")

    # ⑥ Audit
    df["created_at"] = _now()

    return df[[
        "product_id", "product_name",
        "aisle_id", "department_id",
        "aisle_name", "department_name",
        "reorder_rate", "order_frequency",
        "created_at"
    ]]


def transform_dim_user() -> pd.DataFrame:
    """
    Bảng mới — không có trong raw data, phải tổng hợp từ orders.csv.

    Transforms:
      ① total_orders            — tổng số đơn hàng per user
      ② avg_days_between_orders — trung bình số ngày giữa 2 lần mua
                                   (bỏ null = đơn đầu tiên)
      ③ preferred_order_dow     — ngày trong tuần hay mua nhất (mode)
      ④ preferred_order_hour    — giờ hay đặt hàng nhất (mode)
      ⑤ Thêm 'created_at'
    """
    print("    → dim_user  (tổng hợp stats per user từ orders)...")

    # Dùng DuckDB để aggregate nhanh hơn pandas groupby trên file lớn
    orders_path = _parquet("orders")
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            user_id,
            COUNT(*)                                    AS total_orders,
            ROUND(AVG(days_since_prior_order), 2)       AS avg_days_between_orders,
            -- Mode: lấy giá trị xuất hiện nhiều nhất
            MODE(order_dow)                             AS preferred_order_dow,
            MODE(order_hour_of_day)                     AS preferred_order_hour
        FROM read_parquet('{orders_path}')
        GROUP BY user_id
        ORDER BY user_id
    """).df()
    con.close()

    df["avg_days_between_orders"] = df["avg_days_between_orders"].fillna(0)
    df["created_at"] = _now()

    return df[[
        "user_id", "total_orders",
        "avg_days_between_orders",
        "preferred_order_dow", "preferred_order_hour",
        "created_at"
    ]]


def transform_dim_order() -> pd.DataFrame:
    """
    Transforms:
      ① Fill null days_since_prior_order → 0
         (đơn hàng đầu tiên của user chưa có lần mua trước)
      ② Map order_dow (số) → order_day_name (tên thứ)
         0=Sunday, 1=Monday, ..., 6=Saturday
      ③ Thêm cột time_of_day phân loại theo giờ đặt hàng:
         05-11 → morning | 12-16 → afternoon
         17-20 → evening | 21-04 → night
      ④ Thêm 'created_at'
    """
    print("    → dim_order  (fill null + map dow name + time_of_day bucket)...")

    df = _load("orders")

    # ① Fill null
    df["days_since_prior_order"] = df["days_since_prior_order"].fillna(0)

    # ② Map tên thứ
    DOW_MAP = {
        0: "Sunday", 1: "Monday",  2: "Tuesday",
        3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"
    }
    df["order_day_name"] = df["order_dow"].map(DOW_MAP)

    # ③ Phân loại khung giờ
    def _time_of_day(hour: int) -> str:
        if 5  <= hour <= 11: return "morning"
        if 12 <= hour <= 16: return "afternoon"
        if 17 <= hour <= 20: return "evening"
        return "night"

    df["time_of_day"] = df["order_hour_of_day"].apply(_time_of_day)

    # ④ Audit
    df["created_at"] = _now()

    return df[[
        "order_id", "user_id",
        "order_number", "order_dow", "order_day_name",
        "order_hour_of_day", "time_of_day",
        "days_since_prior_order",
        "eval_set", "created_at"
    ]]


# ═══════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

def transform_all() -> dict:
    """
    Chạy tất cả dimension transforms.
    Trả về dict: { table_name: DataFrame }

    Lưu ý: fact_order_items KHÔNG nằm ở đây.
    Fact table được transform + load trực tiếp bằng DuckDB SQL
    trong load.py để tránh load 32M+ dòng vào pandas RAM.
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 — TRANSFORM")
    print("=" * 60)
    print("  Transforming dimension tables...\n")

    transformed = {
        "dim_department": transform_dim_department(),
        "dim_aisle":      transform_dim_aisle(),
        "dim_product":    transform_dim_product(),
        "dim_user":       transform_dim_user(),
        "dim_order":      transform_dim_order(),
    }

    print("\n  Summary:")
    for name, df in transformed.items():
        print(f"    ✓ {name:<20} {len(df):>10,} dòng  ×  {df.shape[1]} cột")

    print("\n  [TRANSFORM] Hoàn tất!\n")
    return transformed
