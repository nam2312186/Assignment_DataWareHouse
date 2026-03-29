"""
=============================================================
Stage 3 — LOAD
=============================================================
Nạp dữ liệu đã transform vào DuckDB Data Warehouse.
Schema: Snowflake Schema

Thứ tự nạp (dependency order):
  1. dim_department
  2. dim_aisle        (FK → dim_department)
  3. dim_product      (FK → dim_aisle, dim_department)
  4. dim_user
  5. dim_order        (FK → dim_user)
  6. fact_order_items (FK → dim_order, dim_product, dim_user)
     → Transform + Load trực tiếp bằng DuckDB SQL từ Parquet
       để tránh load 32M+ dòng vào pandas RAM

Output: data/warehouse/instacart_dw.duckdb
=============================================================
"""

import duckdb
from datetime import datetime
from pathlib import Path

WAREHOUSE_DIR = Path("data/warehouse")
DB_PATH       = WAREHOUSE_DIR / "instacart_dw.duckdb"
STAGING_DIR   = Path("data/staging")

# ── DDL: Snowflake Schema ────────────────────────────────────────────────────
DDL = """
-- ┌──────────────────────────────────────────────────┐
-- │         SNOWFLAKE SCHEMA — INSTACART DW          │
-- └──────────────────────────────────────────────────┘

CREATE TABLE IF NOT EXISTS dim_department (
    department_id    TINYINT  PRIMARY KEY,
    department_name  VARCHAR  NOT NULL,
    created_at       VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_aisle (
    aisle_id         SMALLINT PRIMARY KEY,
    aisle_name       VARCHAR  NOT NULL,
    created_at       VARCHAR
    -- NOTE: FK → dim_department không enforce ở DDL
    --       để tránh lỗi khi load thứ tự không đúng
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_id       INTEGER  PRIMARY KEY,
    product_name     VARCHAR  NOT NULL,
    aisle_id         SMALLINT,
    department_id    TINYINT,
    aisle_name       VARCHAR,
    department_name  VARCHAR,
    reorder_rate     FLOAT,        -- Tính từ prior data (Transform step)
    order_frequency  INTEGER,      -- Tổng số lần mua trong prior
    created_at       VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_user (
    user_id                  INTEGER  PRIMARY KEY,
    total_orders             SMALLINT,
    avg_days_between_orders  FLOAT,
    preferred_order_dow      TINYINT,  -- 0=Sun..6=Sat
    preferred_order_hour     TINYINT,  -- 0..23
    created_at               VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_order (
    order_id                INTEGER  PRIMARY KEY,
    user_id                 INTEGER,
    order_number            SMALLINT,
    order_dow               TINYINT,
    order_day_name          VARCHAR,   -- "Sunday".."Saturday"
    order_hour_of_day       TINYINT,
    time_of_day             VARCHAR,   -- morning/afternoon/evening/night
    days_since_prior_order  FLOAT,     -- null đã fill = 0 (Transform step)
    eval_set                VARCHAR,   -- prior/train/test
    created_at              VARCHAR
);

CREATE TABLE IF NOT EXISTS fact_order_items (
    order_item_id     BIGINT   PRIMARY KEY,
    order_id          INTEGER,
    product_id        INTEGER,
    user_id           INTEGER,
    add_to_cart_order SMALLINT,
    reordered         TINYINT,
    eval_set          VARCHAR,
    created_at        VARCHAR
);
"""

# Bảng và thứ tự nạp
LOAD_ORDER = [
    "dim_department",
    "dim_aisle",
    "dim_product",
    "dim_user",
    "dim_order",
]


def _get_connection() -> duckdb.DuckDBPyConnection:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def _load_dimension(con: duckdb.DuckDBPyConnection, table: str, df) -> int:
    """Insert một dimension DataFrame vào DuckDB table."""
    con.execute(f"DELETE FROM {table}")      # Idempotent: xoá rồi nạp lại
    con.register("_df", df)
    con.execute(f"INSERT INTO {table} SELECT * FROM _df")
    con.unregister("_df")
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return count


def _load_fact_order_items(con: duckdb.DuckDBPyConnection) -> int:
    """
    Transform + Load fact_order_items trực tiếp bằng DuckDB SQL.

    Transforms thực hiện ở đây:
      ① UNION ALL prior + train  (gộp 2 nguồn dữ liệu)
      ② JOIN với orders          (lấy user_id, eval_set)
      ③ ROW_NUMBER()             (tạo surrogate key order_item_id)
      ④ Ghi audit created_at
    """
    prior_path = str(STAGING_DIR / "order_products__prior.parquet")
    train_path = str(STAGING_DIR / "order_products__train.parquet")
    orders_path = str(STAGING_DIR / "orders.parquet")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("    Đang transform + load fact_order_items (32M+ dòng — dùng DuckDB SQL)...")

    con.execute("DELETE FROM fact_order_items")

    con.execute(f"""
        INSERT INTO fact_order_items
        SELECT
            -- ③ Surrogate key: đánh số thứ tự toàn bộ fact table
            ROW_NUMBER() OVER (ORDER BY op.order_id, op.product_id) AS order_item_id,

            op.order_id,
            op.product_id,

            -- ② Lấy user_id từ dim_order (sau khi đã load)
            o.user_id,

            op.add_to_cart_order,
            op.reordered,

            -- ② Lấy eval_set từ orders
            o.eval_set,

            -- ④ Audit timestamp
            '{ts}' AS created_at

        FROM (
            -- ① UNION ALL: gộp prior (32M) + train (1.4M)
            SELECT order_id, product_id, add_to_cart_order, reordered
            FROM read_parquet('{prior_path}')

            UNION ALL

            SELECT order_id, product_id, add_to_cart_order, reordered
            FROM read_parquet('{train_path}')
        ) op

        -- ② Join với orders để lấy user_id và eval_set
        JOIN read_parquet('{orders_path}') o
          ON op.order_id = o.order_id
    """)

    count = con.execute("SELECT COUNT(*) FROM fact_order_items").fetchone()[0]
    return count


def load_all(transformed: dict) -> str:
    """
    Load toàn bộ DW:
      1. Tạo schema (DDL)
      2. Load 5 dimension tables từ transformed DataFrames
      3. Transform + Load fact_order_items bằng DuckDB SQL
    """
    print("\n" + "=" * 60)
    print("  STAGE 3 — LOAD")
    print("=" * 60)
    print(f"  Target DW: {DB_PATH}\n")

    con = _get_connection()

    # Tạo schema
    con.execute(DDL)
    print("  ✓ Snowflake Schema đã tạo xong\n")

    # Load dimension tables
    print("  [Dimensions]")
    for table in LOAD_ORDER:
        df    = transformed[table]
        count = _load_dimension(con, table, df)
        print(f"    ✓ {table:<22} {count:>10,} dòng")

    # Load fact table (Transform bên trong)
    print("\n  [Fact Table]")
    fact_count = _load_fact_order_items(con)
    print(f"    ✓ {'fact_order_items':<22} {fact_count:>10,} dòng")

    con.close()

    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n  DW size: {size_mb:.1f} MB")
    print(f"  [LOAD] Hoàn tất! → {DB_PATH}\n")

    return str(DB_PATH)


def verify_dw() -> None:
    """Kiểm tra nhanh toàn bộ DW sau khi load."""
    print("\n" + "=" * 60)
    print("  VERIFY — Kiểm tra Data Warehouse")
    print("=" * 60)

    con = duckdb.connect(str(DB_PATH), read_only=True)

    tables = [
        "dim_department", "dim_aisle", "dim_product",
        "dim_user", "dim_order", "fact_order_items"
    ]

    for t in tables:
        count   = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        nulls   = con.execute(f"""
            SELECT COUNT(*) FROM {t}
            WHERE {'order_id' if 'fact' in t else list(con.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='{t}' LIMIT 1"
            ).fetchone())} IS NULL
        """)
        print(f"  {t:<25} {count:>12,} dòng")

    # Kiểm tra reorder_rate hợp lệ
    invalid = con.execute(
        "SELECT COUNT(*) FROM dim_product WHERE reorder_rate < 0 OR reorder_rate > 1"
    ).fetchone()[0]
    print(f"\n  reorder_rate ngoài [0,1]: {invalid} (phải = 0)")

    # Kiểm tra null FK trong fact
    null_fk = con.execute(
        "SELECT COUNT(*) FROM fact_order_items WHERE user_id IS NULL"
    ).fetchone()[0]
    print(f"  fact user_id null       : {null_fk} (phải = 0)")

    con.close()
    print("\n  [VERIFY] Hoàn tất!\n")
