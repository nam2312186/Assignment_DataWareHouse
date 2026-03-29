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

    df = con.execute("""
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
    """).df()

    con.close()

    out = MARTS_DIR / "mart_user_item.csv"
    df.to_csv(out, index=False)

    print(f"    ✓ {len(df):,} user-item pairs → {out}")
    print(f"    ✓ {df['user_id'].nunique():,} unique users  |  "
          f"{df['product_id'].nunique():,} unique products")
    print(f"    ✓ Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['product_id'].nunique()):.4%}")
