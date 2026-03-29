"""
=============================================================
Data Mart 1 — Basket Analysis
=============================================================
Mục tiêu: Cung cấp dữ liệu transaction dạng phẳng cho
          thuật toán khai phá luật kết hợp (Apriori / FP-Growth).

Query từ DW (DuckDB) → CSV

Schema output:
  order_id       : ID đơn hàng (= 1 transaction)
  product_name   : Tên sản phẩm trong đơn
  aisle_name     : Danh mục ngách
  department_name: Phòng hàng hóa

File output: data/warehouse/marts/mart_basket.csv
=============================================================
"""

import duckdb
from pathlib import Path

DB_PATH   = Path("data/warehouse/instacart_dw.duckdb")
MARTS_DIR = Path("data/warehouse/marts")


def build_mart_basket() -> None:
    """
    Query DW để tạo mart phục vụ Apriori/FP-Growth.

    Chỉ lấy eval_set = 'prior' (tập training, ~32M interactions)
    vì đây là dữ liệu lịch sử dùng để học luật kết hợp.
    """
    MARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  [MART 1] Building mart_basket (Apriori / FP-Growth)...")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    df = con.execute("""
        SELECT
            f.order_id,
            p.product_name,
            p.aisle_name,
            p.department_name
        FROM fact_order_items  f
        JOIN dim_product       p ON f.product_id = p.product_id
        JOIN dim_order         o ON f.order_id   = o.order_id
        WHERE f.eval_set = 'prior'
        ORDER BY f.order_id, p.product_name
    """).df()

    con.close()

    out = MARTS_DIR / "mart_basket.csv"
    df.to_csv(out, index=False)

    print(f"    ✓ {len(df):,} dòng đã lưu → {out}")
    print(f"    ✓ {df['order_id'].nunique():,} unique orders  |  "
          f"{df['product_name'].nunique():,} unique products")
    print(f"    ✓ Trung bình {len(df) / df['order_id'].nunique():.1f} sản phẩm/đơn")
