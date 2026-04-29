"""
=============================================================
Data Mart 2 — User-Item Interaction
=============================================================
Mục tiêu: Cung cấp ma trận tương tác user-sản phẩm cho
          mô hình đồ thị LightGCN (Graph Neural Network).

Query từ DW (DuckDB) → CSV

Schema output:
  user_id           : ID người dùng (node user trong bipartite graph)
  product_id        : ID sản phẩm  (node product trong bipartite graph)
  product_name      : Tên sản phẩm
  interaction_count : Tổng số lần user mua product này
  reorder_rate      : Tỷ lệ mua lại (AVG(reordered))

File output: data/warehouse/marts/mart_user_item.csv
=============================================================
"""

import duckdb
from pathlib import Path

DB_PATH   = Path("data/warehouse/instacart_dw.duckdb")
MARTS_DIR = Path("data/warehouse/marts")


def build_mart_user_item() -> None:
    """
    Query DW để tạo mart phục vụ LightGCN.

    Aggregate user × product → interaction_count + reorder_rate
    Mỗi cặp (user_id, product_id) là 1 cạnh trong bipartite graph.
    """
    MARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  [MART 2] Building mart_user_item (LightGCN)...")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    out = MARTS_DIR / "mart_user_item.csv"

    # ✅ Dùng DuckDB's native CSV export (tránh load toàn bộ vào RAM)
    con.execute(f"""
        COPY (
            SELECT
                f.user_id,
                f.product_id,
                p.product_name,
                COUNT(*)        AS interaction_count,
                ROUND(AVG(CAST(f.reordered AS DOUBLE)), 4) AS reorder_rate
            FROM fact_order_items  f
            JOIN dim_product       p ON f.product_id = p.product_id
            WHERE f.eval_set = 'prior'
            GROUP BY f.user_id, f.product_id, p.product_name
            ORDER BY f.user_id, interaction_count DESC
        ) TO '{out}' (FORMAT CSV, HEADER TRUE);
    """)

    # ── Get statistics ──
    stats = con.execute("""
        SELECT
            COUNT(*) as pair_count,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT product_id) as unique_products
        FROM (
            SELECT
                f.user_id,
                f.product_id
            FROM fact_order_items  f
            WHERE f.eval_set = 'prior'
            GROUP BY f.user_id, f.product_id
        )
    """).fetchall()

    con.close()

    pair_count, unique_users, unique_products = stats[0]
    sparsity = 1 - pair_count / (unique_users * unique_products)

    print(f"    ✓ {pair_count:,} user-item pairs → {out}")
    print(f"    ✓ {unique_users:,} unique users  |  "
          f"{unique_products:,} unique products")
    print(f"    ✓ Sparsity: {sparsity:.4%}")
