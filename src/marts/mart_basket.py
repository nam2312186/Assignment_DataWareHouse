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

    out = MARTS_DIR / "mart_basket.csv"
    
    # ✅ Dùng DuckDB's native CSV export (tránh load toàn bộ vào RAM)
    con.execute(f"""
        COPY (
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
        ) TO '{out}' (FORMAT CSV, HEADER TRUE);
    """)

    # ──  Get statistics (load chỉ một lần cho thống kê) ──
    stats = con.execute("""
        SELECT
            COUNT(*) as row_count,
            COUNT(DISTINCT order_id) as unique_orders,
            COUNT(DISTINCT product_name) as unique_products
        FROM (
            SELECT
                f.order_id,
                p.product_name
            FROM fact_order_items  f
            JOIN dim_product       p ON f.product_id = p.product_id
            JOIN dim_order         o ON f.order_id   = o.order_id
            WHERE f.eval_set = 'prior'
        )
    """).fetchall()

    con.close()

    row_count, unique_orders, unique_products = stats[0]

    print(f"    ✓ {row_count:,} dòng đã lưu → {out}")
    print(f"    ✓ {unique_orders:,} unique orders  |  "
          f"{unique_products:,} unique products")
    print(f"    ✓ Trung bình {row_count / unique_orders:.1f} sản phẩm/đơn")
