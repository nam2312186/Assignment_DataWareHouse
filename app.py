import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Instacart Data Warehouse",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIG PATHS ---
DB_PATH = Path("data/warehouse/instacart_dw.duckdb")
MART_BASKET = Path("data/warehouse/marts/mart_basket.csv")

# --- CACHE DATA FUNCTIONS ---
def get_db_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)

@st.cache_data
def load_overview_metrics():
    with get_db_connection() as con:
        metrics = con.execute("""
            SELECT 
                (SELECT COUNT(*) FROM dim_order) as total_orders,
                (SELECT COUNT(*) FROM dim_user) as total_users,
                (SELECT COUNT(*) FROM dim_product) as total_products,
                (SELECT COUNT(*) FROM fact_order_items) as total_interactions,
                (SELECT COUNT(*) FROM fact_order_items) * 1.0 / (SELECT COUNT(*) FROM dim_order) as avg_basket_size
        """).fetchone()
        return metrics

@st.cache_data
def load_treemap_data():
    with get_db_connection() as con:
        df = con.execute("""
            SELECT 
                d.department_name,
                a.aisle_name,
                COUNT(*) as items_sold
            FROM fact_order_items f
            JOIN dim_product p ON f.product_id = p.product_id
            JOIN dim_department d ON p.department_id = d.department_id
            JOIN dim_aisle a ON p.aisle_id = a.aisle_id
            GROUP BY d.department_name, a.aisle_name
            ORDER BY items_sold DESC
        """).df()
        return df

@st.cache_data
def load_raw_data(limit=100, dept=None):
    with get_db_connection() as con:
        query = """
            SELECT 
                f.order_id,
                u.user_id,
                p.product_name,
                d.department_name,
                a.aisle_name,
                o.order_hour_of_day,
                f.add_to_cart_order,
                f.reordered
            FROM fact_order_items f
            JOIN dim_product p ON f.product_id = p.product_id
            JOIN dim_order o ON f.order_id = o.order_id
            JOIN dim_user u ON o.user_id = u.user_id
            JOIN dim_department d ON p.department_id = d.department_id
            JOIN dim_aisle a ON p.aisle_id = a.aisle_id
        """
        if dept and dept != "Tất Cả":
            query += f" WHERE d.department_name = '{dept}'"
        query += f" LIMIT {limit}"
        return con.execute(query).df()

@st.cache_data
def load_top_departments():
    with get_db_connection() as con:
        df = con.execute("""
            SELECT 
                d.department_name,
                COUNT(*) as items_sold
            FROM fact_order_items f
            JOIN dim_product p ON f.product_id = p.product_id
            JOIN dim_department d ON p.department_id = d.department_id
            GROUP BY d.department_name
            ORDER BY items_sold DESC
            LIMIT 10
        """).df()
        return df

@st.cache_data
def load_order_hours():
    with get_db_connection() as con:
        df = con.execute("""
            SELECT 
                order_hour_of_day,
                COUNT(*) as total
            FROM dim_order
            GROUP BY order_hour_of_day
            ORDER BY order_hour_of_day
        """).df()
        return df

@st.cache_data
def load_dow_distribution():
    with get_db_connection() as con:
        df = con.execute("""
            SELECT 
                order_dow, 
                COUNT(*) as total_orders 
            FROM dim_order 
            GROUP BY order_dow 
            ORDER BY order_dow
        """).df()
        days = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        df['day_name'] = df['order_dow'].map(days)
        return df

@st.cache_data
def load_days_since_prior():
    with get_db_connection() as con:
        df = con.execute("""
            SELECT 
                days_since_prior_order, 
                COUNT(*) as total_orders 
            FROM dim_order 
            WHERE days_since_prior_order IS NOT NULL
            GROUP BY days_since_prior_order 
            ORDER BY days_since_prior_order
        """).df()
        return df

@st.cache_data
def load_reorder_rates():
    with get_db_connection() as con:
        df = con.execute("""
            SELECT 
                d.department_name,
                AVG(p.reorder_rate) as avg_reorder_rate
            FROM dim_product p
            JOIN dim_department d ON p.department_id = d.department_id
            WHERE p.reorder_rate > 0 AND p.order_frequency > 100
            GROUP BY d.department_name
            ORDER BY avg_reorder_rate DESC
        """).df()
        
        # Overall reorder rate
        overall = con.execute("""
            SELECT SUM(CASE WHEN reordered = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) 
            FROM fact_order_items
        """).fetchone()[0]
        return df, overall

@st.cache_data
def load_product_scatter():
    with get_db_connection() as con:
        df = con.execute("""
            SELECT 
                p.product_name,
                d.department_name,
                p.reorder_rate,
                p.order_frequency
            FROM dim_product p
            JOIN dim_department d ON p.department_id = d.department_id
            WHERE p.order_frequency > 500
        """).df()
        return df

@st.cache_data
def load_department_products(dept_name):
    with get_db_connection() as con:
        df = con.execute(f"""
            SELECT 
                p.product_name,
                p.reorder_rate,
                p.order_frequency
            FROM dim_product p
            JOIN dim_department d ON p.department_id = d.department_id
            WHERE d.department_name = '{dept_name}' AND p.order_frequency > 500
            ORDER BY p.reorder_rate DESC
            LIMIT 15
        """).df()
        return df

@st.cache_data
def train_association_rules(top_n=500):
    if not MART_BASKET.exists():
        return None
    
    df = pd.read_csv(MART_BASKET)
    
    # Lấy top N sản phẩm phổ biến nhất để sinh luật nhanh trong RAM
    top_products = df['product_name'].value_counts().nlargest(top_n).index
    df_filtered = df[df['product_name'].isin(top_products)].copy()
    
    # Lấy 100,000 transactions ngẫu nhiên từ mart để sinh luật thật nhanh trên Streamlit 
    # (Đủ để demo tính khả thi của hệ thống thời gian thực)
    sample_orders = df_filtered['order_id'].drop_duplicates().sample(n=min(100000, df_filtered['order_id'].nunique()), random_state=42)
    df_sample = df_filtered[df_filtered['order_id'].isin(sample_orders)]
    
    transactions = df_sample.groupby('order_id')['product_name'].apply(list).tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    basket_sets = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    freq_items = fpgrowth(basket_sets, min_support=0.005, use_colnames=True)
    rules = association_rules(freq_items, metric='lift', min_threshold=1.1)
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    return rules

def predict_next_items(current_basket, rules_df, top_n=5):
    if rules_df is None or rules_df.empty: return []
    basket_set = set(current_basket)
    
    matched = rules_df[rules_df['antecedents'].apply(lambda x: x.issubset(basket_set))]
    matched = matched.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    
    suggested = []
    for item_set in matched['consequents']:
        for item in item_set:
            if item not in basket_set and item not in suggested:
                suggested.append(item)
                if len(suggested) == top_n:
                    return suggested
    return suggested


# --- SIDEBAR NAV ---
st.sidebar.title("🛒 Instacart DW")
st.sidebar.info("Dashboard khai phá dữ liệu đồ án Data Warehouse - 32 Triệu Rows (DuckDB)")
page = st.sidebar.radio("Điều hướng", ["0. Kiến trúc & ETL", "1. Tổng Quan DW", "2. Reorder Analytics", "3. Khai Phá & Gợi Ý"])

if not DB_PATH.exists():
    st.error(f"❌ Không tìm thấy Data Warehouse tại {DB_PATH}. Hãy chạy thư mục ETL trước!")
    st.stop()


# ==========================================
# PAGE 0: ARCHITECTURE & ETL
# ==========================================
if page == "0. Kiến trúc & ETL":
    st.title("🏗️ Sơ Đồ Kiến Trúc Hệ Thống (Data Architecture)")
    st.markdown("Quy trình biến đổi dữ liệu từ dạng CSV thô (Raw) thành kho dữ liệu phục vụ Machine Learning.")
    
    st.header("1. Luồng xử lý ETL (Extract - Transform - Load)")
    etl_graph = '''
    digraph ETL {
        rankdir=LR;
        node [shape=box, style=filled, fontname="Helvetica"];
        
        csv [label="Raw Data\\n(CSV Files)", shape=folder, fillcolor="#FFECB3"];
        staging [label="Staging Area\\n(Parquet Files)", shape=folder, fillcolor="#FFECB3"];
        dw [label="Data Warehouse\\n(DuckDB)", shape=cylinder, fillcolor="#BBDEFB"];
        mart1 [label="Data Mart: Basket\\n(mart_basket.csv)", shape=note, fillcolor="#C8E6C9"];
        mart2 [label="Data Mart: User-Item\\n(mart_user_item.csv)", shape=note, fillcolor="#C8E6C9"];
        
        extract [label="Extract\\n(Type Casting)", shape=rarrow, fillcolor="#FFE082"];
        transform [label="Transform\\n(Null Fill, Enrichment)", shape=rarrow, fillcolor="#FFE082"];
        load [label="Load\\n(SQL Union,\\nSurrogate Keys)", shape=rarrow, fillcolor="#FFE082"];
        mining [label="Data Mining Models\\n(FP-Growth & LightGCN)", shape=component, fillcolor="#E1BEE7"];

        csv -> extract;
        extract -> staging;
        staging -> transform;
        transform -> load;
        load -> dw;
        dw -> mart1 [label=" SQL Query"];
        dw -> mart2 [label=" SQL Query"];
        mart1 -> mining [label=" Association Rules"];
        mart2 -> mining [label=" Bipartite Graph"];
    }
    '''
    st.graphviz_chart(etl_graph, use_container_width=True)
    
    st.divider()
    
    st.header("2. Data Warehouse Schema (Snowflake)")
    st.markdown("Kho dữ liệu DuckDB được tổ chức với 1 bảng Fact bao quanh bởi 4 bảng Dimension (phân cấp tại Aisle và Department).")
    schema_graph = '''
    digraph G {
        rankdir=LR;
        nodesep=0.5;
        ranksep=0.8;
        node [shape=record, style=filled, fontname="Helvetica", color="#424242"];
        edge [color="#757575", penwidth=1.5];

        fact [label="{fact_order_items|order_id (FK)\\nproduct_id (FK)\\nadd_to_cart_order\\nreordered}", fillcolor="#FFCCBC"];
        
        dim_order [label="{dim_order|order_id (PK)\\nuser_id (FK)\\norder_number\\n...}", fillcolor="#B2DFDB"];
        dim_user [label="{dim_user|user_id (PK)\\ntotal_orders\\n...}", fillcolor="#B2DFDB"];
        
        dim_product [label="{dim_product|product_id (PK)\\nproduct_name\\naisle_id (FK)\\ndepartment_id (FK)\\n...}", fillcolor="#B2DFDB"];
        dim_aisle [label="{dim_aisle|aisle_id (PK)\\naisle}", fillcolor="#B2DFDB"];
        dim_department [label="{dim_department|department_id (PK)\\ndepartment}", fillcolor="#B2DFDB"];

        // Trục trái (User -> Order -> Fact)
        dim_user -> dim_order [label=" 1:N" dir=forward];
        dim_order -> fact [label=" 1:N" dir=forward];
        
        // Trục phải (Fact -> Product -> Aisle/Dept) dùng dir=back để giữ chiều mũi tên từ Dim -> Fact
        fact -> dim_product [label=" 1:N" dir=back];
        dim_product -> dim_aisle [label=" 1:N" dir=back];
        dim_product -> dim_department [label=" 1:N" dir=back];
    }
    '''
    c1, c2, c3 = st.columns([0.5, 4, 0.5])
    with c2:
        st.graphviz_chart(schema_graph, use_container_width=True)


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
elif page == "1. Tổng Quan DW":
    st.title("📊 Tổng Quan Kho Dữ Liệu")
    st.markdown("Truy cập và truy vấn (Query) trực tiếp khối dữ liệu 32 triệu dòng từ `instacart_dw.duckdb`")
    
    metrics = load_overview_metrics()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Tổng Quy mô (Facts)", f"{metrics[3]:,} Dòng", '+ OLAP DuckDB')
    with col2: st.metric("Tổng Doanh Thu (Đơn)", f"{metrics[0]:,}")
    with col3: st.metric("Tổng Người Dùng", f"{metrics[1]:,}")
    with col4: st.metric("Danh Mục Sản Phẩm", f"{metrics[2]:,}")
    with col5: st.metric("Kích cỡ Giỏ/Đơn", f"{metrics[4]:.1f} SP")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏬 Thị Phần (Treemap)", "📈 Lưu lượng Khách", "🔄 Chu Kỳ Mua", "🔎 Tra cứu Raw Data"])
    
    with tab1:
        st.subheader("Phân bổ thị phần theo Ngành Hàng & Quầy Hàng (Department -> Aisle)")
        df_tree = load_treemap_data()
        fig_tree = px.treemap(
            df_tree, 
            path=['department_name', 'aisle_name'], 
            values='items_sold',
            color='items_sold',
            color_continuous_scale='Portland',
            title='Treemap Thị phần (Click để phóng to & thu nhỏ)'
        )
        fig_tree.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=600)
        st.plotly_chart(fig_tree, use_container_width=True)
        
    with tab2:
        st.subheader("Lưu lượng đặt hàng Instacart cực đại")
        c21, c22 = st.columns(2)
        with c21:
            df_hours = load_order_hours()
            fig2 = px.area(
                df_hours, x='order_hour_of_day', y='total', markers=True, 
                title="Khung Giờ Mua Sắm (Area Chart)",
                color_discrete_sequence=['#FF7F50']
            )
            st.plotly_chart(fig2, use_container_width=True)
        with c22:
            df_dow = load_dow_distribution()
            fig3 = px.bar(
                df_dow, x='day_name', y='total_orders', 
                title="Lưu Lượng Theo Ngày (Bar Chart)",
                color='total_orders', color_continuous_scale='Sunset'
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Phân bố số ngày khách hàng MỚI QUAY LẠI đặt đơn")
        st.markdown("Đỉnh nhọn ở khoảng 7 ngày (1 tuần) và 30 ngày (1 tháng) phản ánh đúng thói quen đi siêu thị (đi hàng tuần hoặc hàng tháng).")
        df_days = load_days_since_prior()
        fig_days = px.bar(
            df_days, x='days_since_prior_order', y='total_orders',
            title="Sự phân cực về chu kỳ mua hàng (0-30 days)",
            color_discrete_sequence=['#4DB6AC']
        )
        st.plotly_chart(fig_days, use_container_width=True)
        
    with tab4:
        st.subheader("Bảng Dữ Liệu Thô (Interactive Dataframe)")
        st.markdown("Tra cứu và tuỳ biến bảng dữ liệu khổng lồ (kết hợp tự động nhờ SQL Join).")
        
        # Interactive filters
        df_dept_list = load_top_departments()
        depts = ["Tất Cả"] + df_dept_list['department_name'].tolist()
        
        c1, c2 = st.columns([1, 3])
        with c1: filter_dept = st.selectbox("Lọc theo Department:", depts)
        with c2: filter_limit = st.slider("Giới hạn số dòng hiển thị (Limit):", min_value=10, max_value=10000, value=100, step=50)
        
        with st.spinner("DuckDB đang truy vấn dữ liệu theo thời gian thực..."):
            raw_data = load_raw_data(limit=filter_limit, dept=filter_dept)
            
        st.dataframe(raw_data, use_container_width=True, height=450)


# ==========================================
# PAGE 2: REORDER ANALYTICS
# ==========================================
elif page == "2. Reorder Analytics":
    st.title("🔁 Phân Tích Lòng Trung Thành (Reorder Analytics)")
    st.markdown("Đo lường mức độ 'trung thành' của khách hàng dựa trên **Tỷ lệ mua lại (Reorder Rate)**.")
    
    df_dept_ro, overall_ro = load_reorder_rates()
    
    col1, col2, col3 = st.columns(3)
    best_dept = df_dept_ro.iloc[0]['department_name']
    best_rate = df_dept_ro.iloc[0]['avg_reorder_rate']
    with col1: st.metric("Tổng Tỉ lệ Mua Lại (Toàn Hệ Thống)", f"{overall_ro:.1%}", "Từ 32M Dòng")
    with col2: st.metric("Ngành Hàng Giữ Chân Tốt Nhất", best_dept)
    with col3: st.metric("Tỷ Lệ Trung Bình của Ngành", f"{best_rate:.1%}")
    
    st.divider()
    
    st.subheader("Mối tương quan giữa Độ Phổ Biến và Mức Gây Nghiện (Loyalty)")
    st.markdown("Biểu đồ phân tán (Scatter Plot) kết xuất tự động từ DuckDB (Scale trục x theo logarithm).")
    df_scatter = load_product_scatter()
    fig_scatter = px.scatter(
        df_scatter, x='order_frequency', y='reorder_rate', 
        color='department_name', hover_name='product_name',
        log_x=True,
        title="Đám Mây Tương Quan Sản Phẩm (Hover để xem tên SP)",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.divider()
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("Xếp Hạng Ngành Hàng (Avg Reorder)")
        fig = px.bar(df_dept_ro, x='avg_reorder_rate', y='department_name', orientation='h', 
                     color='avg_reorder_rate', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Khám phá Sản Phẩm 'Mỏ Vàng' (Interactive)")
        st.markdown("Lọc ra các sản phẩm xuất sắc nhất mang lại doanh thu lặp vòng.")
        
        selected_dept = st.selectbox("Chọn Department cần khảo sát:", df_dept_ro['department_name'].tolist())
        
        with st.spinner("DuckDB đang truy quét sản phẩm..."):
            df_top_prod = load_department_products(selected_dept)
            
        fig2 = px.bar(df_top_prod, x='reorder_rate', y='product_name', orientation='h',
                      text=[f"{x:.1%}" for x in df_top_prod['reorder_rate']],
                      title=f"Top 15 Sản Phẩm Gây Nghiện Trong {selected_dept}",
                      color='order_frequency', color_continuous_scale='Plasma')
        fig2.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='.1%')
        st.plotly_chart(fig2, use_container_width=True)


# ==========================================
# PAGE 3: ASSOCIATION RULES
# ==========================================
elif page == "3. Khai Phá & Gợi Ý":
    st.title("🛒 Khai Phá & Gợi Ý Sản Phẩm")
    
    st.markdown("So sánh hai phương pháp khuyến nghị (Recommendation Algorithm) đã được huấn luyện trên Data Warehouse:")
    
    import os
    import json
    
    # Đọc kết quả tự động từ file sau khi chạy benchmark Terminal
    metrics_path = os.path.join("evaluation", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            df_eval = pd.DataFrame({
                "Tiêu chí": ["Điểm Recall (%)", "Điểm Precision (%)", "Điểm F1-Score (%)"],
                "FP-Growth (Luật kết hợp)": [data.get("recall_fp", 0), data.get("precision_fp", 0), data.get("f1_fp", 0)],
                "LightGCN (Học sâu đồ thị)": [data.get("recall_gnn", 0), data.get("precision_gnn", 0), data.get("f1_gnn", 0)]
            })
    else:
        # Chưa chạy Model, Set Null
        df_eval = pd.DataFrame({
            "Tiêu chí": ["Điểm Recall (%)", "Điểm Precision (%)", "Điểm F1-Score (%)"],
            "FP-Growth (Luật kết hợp)": [None, None, None],
            "LightGCN (Học sâu đồ thị)": [None, None, None]
        })
        
    df_melted = pd.melt(df_eval, id_vars='Tiêu chí', var_name='Thuật toán', value_name='Điểm số (%)')
    
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if os.path.exists(metrics_path):
            fig_perf = px.bar(df_melted, x='Thuật toán', y='Điểm số (%)', color='Tiêu chí', 
                              barmode='group', title="Tương Quan Độ Chính Xác (Càng cao càng tốt)",
                              color_discrete_sequence=['#FF9800', '#5C6BC0'])
            fig_perf.update_layout(
                height=320, 
                margin=dict(t=50, b=10, l=10, r=10),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01) # Đẩy legend ra ngoài để không đè title
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.warning("⚠️ **VÙNG CHỜ DỮ LIỆU ĐÁNH GIÁ (NULL)**")
            st.info("Để biểu đồ và bảng kết quả hiển thị, bạn cần mở Terminal (Cmd/PowerShell), sau đó khởi động kịch bản huấn luyện bằng lệnh:\n\n 💻 `python evaluate_models.py`")
            
    with c2:
        st.subheader("Bảng Đánh Giá Offline")
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
        st.info("💡 **Kết luận**: GNN (LightGCN) vượt trội do học được mạng lưới ẩn của đồ thị User-Item thay vì đếm luật cứng nhắc. Bù lại, FP-Growth cực kì nhẹ, dễ triển khai realtime (O(n)).")
        
    st.divider()
    
    st.subheader("⚡ Demo Live Cỗ Máy Bán Chéo (Cross-Selling)")
    st.markdown("Khởi tạo **FP-Growth** bằng dữ liệu trích xuất từ Data Mart `mart_basket.csv` ở backend.")
    st.caption("*(❓ **Tại sao Web App chỉ dùng thuật toán FP-Growth?** Vì xử lý đồ thị Mạng Neural (LightGCN) tốn hàng giờ Training nạp ma trận khổng lồ. Ngược lại, thuật toán FP-Growth sinh Luật Siêu Nhẹ (chạy mượt trên RAM), tra cứu O(1) nên cực kỳ hoàn hảo cho thao tác tương tác thời gian thực trên Web)*")
    
    with st.spinner("Đang huấn luyện/chạy thuật toán FP-Growth từ Data Mart (Chạy lần đầu mất ~10s)..."):
        rules = train_association_rules(top_n=800)
    
    if rules is None:
        st.error("❌ Không tìm thấy file `mart_basket.csv`. Vui lòng chạy ETL pipeline để build Data Mart.")
        st.stop()
        
    st.success(f"✅ Hoàn tất tải Data Mart. Khai phá thành công {len(rules):,} luật kết hợp (Association Rules).")
    
    # Tạo danh sách các sản phẩm có thể gợi ý
    all_antecedents = set()
    for item_set in rules['antecedents']:
        all_antecedents.update(list(item_set))
    all_antecedents = sorted(list(all_antecedents))
    
    st.divider()
    
    st.subheader("Thử Nghiệm Gợi Ý Khách Hàng (Next-Item Prediction)")
    selected_items = st.multiselect(
        "Khách hàng hiện đang chọn mua món gì vào giỏ? (Có thể chọn nhiều)",
        options=all_antecedents,
        default=["Banana"] if "Banana" in all_antecedents else []
    )
    
    if selected_items:
        with st.spinner("Đang lục tìm siêu dữ liệu..."):
            suggestions = predict_next_items(selected_items, rules, top_n=5)
            
        if suggestions:
            st.markdown("### ✨ Gợi Ý Bán Chéo (Cross-Sell) Thành Công!")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                cols[i].info(f"{i+1}. **{suggestion}**")
                
            st.markdown("#### Chi tiết các Luật (Rules) đang áp dụng:")
            basket_set = set(selected_items)
            matched = rules[rules['antecedents'].apply(lambda x: x.issubset(basket_set))]
            st.dataframe(
                matched[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10),
                use_container_width=True
            )
        else:
            st.warning("⚠️ Chưa phát hiện được luật kết hợp/Gợi ý nào đủ mạnh cho kết hợp bạn vừa chọn!")

