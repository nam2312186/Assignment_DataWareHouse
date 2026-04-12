import os
import pandas as pd
import time
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv() # Khởi chạy và nạp các biến số từ file .env

import random
import numpy as np
import torch
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from src.eval_utils import create_holdout_test_set, calculate_hit_rate, predict_apriori
from src.gnn_utils import prepare_graph_data, train_lightgcn, predict_gnn_bipartite

def seed_everything(seed=42):
    """Máy sẽ luôn cho ra 1 kết quả duy nhất để dễ làm báo cáo"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_evaluation():
    # Chốt hạt giống ngay khi bắt đầu
    seed_everything(42)
    
    # ----------------------------------------------------
    # Chuẩn bị Data
    # ----------------------------------------------------
    print("="*65)
    print("🛒 INSTACART DATA WAREHOUSE - AI MODELS EVALUATION (TERMINAL)")
    print("="*65)
    
    csv_path = os.path.join("data", "warehouse", "marts", "mart_basket.csv")
    print("\n[1] Đang nạp Data Mart (mart_basket.csv)...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy {csv_path}. Hãy chạy python run_etl.py trước.")
        return
        
    print(f"✅ Đã tải {len(df):,} dòng dữ liệu.")
    
    # Lấy linh động lượng Data gốc từ thư viện .env
    sample_ratio = float(os.getenv("SAMPLE_DATA_RATIO", 0.05))
    total_unique_orders = df['order_id'].nunique()
    target_orders = int(total_unique_orders * sample_ratio)
    
    print(f"\n[1.5] Lấy mẫu (Random Sampling) tròn {sample_ratio*100}% Data: {target_orders:,} đơn hàng...")
    
    sample_orders = df['order_id'].drop_duplicates().sample(n=target_orders, random_state=42)
    df_filtered = df[df['order_id'].isin(sample_orders)].copy()
    
    # Để đồ thị (Graph) không bị loãng bởi các sản phẩm "Rác" (Chỉ ai mua 1 lần rồi thôi)
    # Ta giữ lại các mặt hàng phổ thông nhất (Top 2000 mặt hàng)
    top_products = df_filtered['product_name'].value_counts().nlargest(2000).index
    df_filtered = df_filtered[df_filtered['product_name'].isin(top_products)]
    
    print("\n[2] Đang tạo tập Train / Test (Hold-out Validation)...")
    train_df, test_cases = create_holdout_test_set(df_filtered, n_test_orders=1000)
    print(f"✅ Dữ liệu để Học (Train set): {len(train_df['order_id'].unique()):,} đơn hàng.")
    print(f"✅ Dữ liệu để Thi (Test set): {len(test_cases):,} đơn hàng.")
    
    
    # ----------------------------------------------------
    # Model 1: FP-Growth
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("🚀 THUẬT TOÁN 1: FP-GROWTH (LUẬT KẾT HỢP)")
    print("="*50)
    
    print("-> Bước 1/2: Bắt đầu Training (Sinh tập phổ biến)...")
    start_time = time.time()
    
    transactions = train_df.groupby('order_id')['product_name'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    basket_sets = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    freq_items = fpgrowth(basket_sets, min_support=0.005, use_colnames=True)
    rules = association_rules(freq_items, metric='lift', min_threshold=1.1)
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    
    fp_train_time = time.time() - start_time
    print(f"✅ Train xong! Dịch ra được {len(rules):,} luật. (Mất: {fp_train_time:.2f} s)")
    
    print("-> Bước 2/2: Quá trình Dự đoán & Đánh giá (Evaluation)...")
    _, _, fp_precision, fp_recall = calculate_hit_rate(
        test_cases, 
        model_type="FP-Growth",
        model_predictor=predict_apriori,
        rules=rules,
        k=20
    )
    print(f"   - Tỉ lệ bắt trúng (Recall):    {fp_recall:.2%}")
    print(f"   - Độ chính xác (Precision):    {fp_precision:.2%}")
    fp_f1 = (2 * fp_precision * fp_recall) / (fp_precision + fp_recall) if (fp_precision + fp_recall) > 0 else 0
    print(f"   - Điểm F1-Score tổng hợp:      {fp_f1:.2%}")
    
    
    # ----------------------------------------------------
    # Model 2: LightGCN
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("🚀 THUẬT TOÁN 2: LIGHTGCN (HỌC SÂU ĐỒ THỊ)")
    print("="*50)
    
    try:
        import torch
        from torch_geometric.nn import LightGCN
    except ImportError:
        print("❌ Lỗi: Máy tính vẫn đang cài đặt thư viện 'torch' và 'torch_geometric' (Hơn 2GB).")
        print("👉 Đừng nóng vội! Hãy chờ lệnh PIP cài đặt xong 100% rồi chạy lại file này.")
        return
        
    print("-> Bước 1/3: Xây dựng đồ thị lưỡng phân (Bipartite Graph)...")
    all_products = df_filtered['product_name'].unique()
    edge_index, metadata = prepare_graph_data(train_df, all_products)
    print(f"✅ Đồ thị tạo xong: {metadata['num_nodes']:,} Nodes, {edge_index.shape[1]:,} Edges.")
    
    print("-> Bước 2/3: Bắt đầu Training LightGCN (Graph Convolution) x 20 Epochs...")
    model, item_emb, gnn_time = train_lightgcn(edge_index, metadata, epochs=20, dim=32)
    print(f"✅ Train xong! (Thời gian tính toán mạng Neural: {gnn_time:.2f} s)")
    
    print("-> Bước 3/3: Quá trình Dự đoán & Đánh giá (Evaluation)...")
    def gnn_predictor(input_basket, **kwargs):
        return predict_gnn_bipartite(input_basket, item_emb, metadata, k=kwargs.get('k', 20))
        
    _, _, gnn_precision, gnn_recall = calculate_hit_rate(
        test_cases, 
        model_type="LightGCN",
        model_predictor=gnn_predictor,
        k=20
    )
    print(f"📊 Kết quả LightGCN (K=20):")
    print(f"   - Tỉ lệ bắt trúng (Recall):    {gnn_recall:.2%}")
    print(f"   - Độ chính xác (Precision):    {gnn_precision:.2%}")
    gnn_f1 = (2 * gnn_precision * gnn_recall) / (gnn_precision + gnn_recall) if (gnn_precision + gnn_recall) > 0 else 0
    print(f"   - Điểm F1-Score tổng hợp:      {gnn_f1:.2%}")

    # ----------------------------------------------------
    # IN BẢNG TỔNG KẾT VÀ GHI FILE REPORT
    # ----------------------------------------------------
    report_text = "="*65 + "\n"
    report_text += "🏆 BẢNG KẾT LUẬN SO SÁNH TRỰC TIẾP (OFFLINE BENCHMARK)\n"
    report_text += "="*65 + "\n"
    report_text += f"{'Tiêu chí (Metric)':<20} | {'FP-Growth':<15} | {'LightGCN (GNN)':<15}\n"
    report_text += "-" * 59 + "\n"
    report_text += f"{'Chỉ số Recall':<20} | {fp_recall*100:>13.2f}% | {gnn_recall*100:>13.2f}%\n"
    report_text += f"{'Chỉ số Precision':<20} | {fp_precision*100:>13.2f}% | {gnn_precision*100:>13.2f}%\n"
    report_text += f"{'Điểm F1-Score':<20} | {fp_f1*100:>13.2f}% | {gnn_f1*100:>13.2f}%\n"
    report_text += f"{'Tốc độ Training':<20} | {fp_train_time:>13.2f}s | {gnn_time:>13.2f}s\n"
    
    print("\n" + report_text)
    
    # Xuất file báo cáo text (.txt)
    os.makedirs("evaluation", exist_ok=True)
    report_path = os.path.join("evaluation", "benchmark_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n💡 Phân tích: Thuật toán AI LightGCN có khả năng tìm kiếm món hàng rộng hơn, ")
        f.write("do đó Recall cao hơn gấp nhiều lần. Nhưng đánh đổi lại tốc độ Train chậm hơn đáng kể.")
        
    # Xuất data tĩnh (.json) để Streamlit đọc lên giao diện tự động
    import json
    json_path = os.path.join("evaluation", "metrics.json")
    metrics_data = {
        "recall_fp": round(fp_recall * 100, 2),
        "precision_fp": round(fp_precision * 100, 2),
        "f1_fp": round(fp_f1 * 100, 2),
        "recall_gnn": round(gnn_recall * 100, 2),
        "precision_gnn": round(gnn_precision * 100, 2),
        "f1_gnn": round(gnn_f1 * 100, 2)
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"✅ Đã lưu kết quả bài test vào file: {report_path} và {json_path}")
    print("🚀 CHƯƠNG TRÌNH KẾT THÚC!\n")

if __name__ == "__main__":
    run_evaluation()
