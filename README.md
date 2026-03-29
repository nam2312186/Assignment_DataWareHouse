# 🛒 Instacart Market Basket Analysis & Data Warehouse

Dự án này là một phiên bản nâng cấp toàn diện (End-to-End) dành cho môn học **Data Warehouse (Kho Dữ Liệu)** và Khai phá Dữ liệu. Hệ thống xây dựng một luồng xử lý (ETL Pipeline) chuyên nghiệp biến đổi tập dữ liệu **32 triệu giao dịch** (từ Instacart) thành một mô hình **Snowflake Schema** mạnh mẽ, phục vụ trực tiếp cho các thuật toán Data Mining và giao diện Dashboard thời gian thực.

---

## 🏗️ Kiến Trúc Hệ Thống (Architecture)

Toàn bộ dự án được thiết kế theo mô hình kiến trúc Dữ liệu hiện đại, chia làm 3 phân hệ chính:

1. **ETL Pipeline & Data Warehouse (`src/etl/`)**
   - **Extract**: Đọc và gom nhóm dữ liệu thô (từ các file `data/raw/*.csv` nặng gigabytes), phân vùng (chunking) xử lý thành định dạng `.parquet` nén siêu cấp tại khu vực Staging.
   - **Transform & Load**: Đẩy toàn bộ dữ liệu qua cỗ máy OLAP **[DuckDB](https://duckdb.org/)**, sử dụng các truy vấn SQL siêu tốc để tạo ra hệ CSDL phân cấp **Snowflake** (1 bảng Fact `fact_order_items` trung tâm và 4 bảng Dimension vệ tinh: `dim_order`, `dim_user`, `dim_product`, liên kết nhánh đa tầng `dim_aisle`, `dim_department`).
2. **Data Marts (`src/marts/`)**
   - Tự động vắt xuất các khung dữ liệu đặc thù (Data Marts) đã tối ưu chuẩn bị cho huấn luyện Mô Hình Khai Phá (`mart_basket.csv` cho luật kết hợp và `mart_user_item.csv` cho GNN).
3. **Machine Learning (`notebook/`)**
   - Huấn luyện thuật toán tìm kiếm món hàng nên mua (Market Basket) sử dụng bộ đôi kinh điển **FP-Growth (Apriori)** và thuật toán học sâu thị giác đồ thị **LightGCN**. Phân tích sâu hành vi mua sắm.
4. **Interactive Dashboard (`app.py`)**
   - Ứng dụng **Streamlit** trực quan kết nối Online (Live) vào Kho DuckDB giúp vẽ ra hàng chục biểu đồ Interactive tuyệt đẹp về thị phần, thời gian mua sắm, tái mua và cung cấp Demo chạy luật kết hợp Recommend món hàng ngay trên Frontend.

---

## 🚀 Cấu Trúc Thư Mục

```text
Instacart-Market-Basket-Analysis/
├── app.py                   # 🔥 (MAIN) Khởi chạy giao diện Streamlit Dashboard
├── run_etl.py               # 🔥 (MAIN) Khởi chạy quy trình ETL tạo Kho Dữ Liệu
├── requirements.txt         # Các thư viện và module phụ thuộc
├── data/
│   ├── raw/                 # Nơi chứa các tệp CSV Kaggle ban đầu
│   ├── staging/             # Dữ liệu tạm .parquet trong quá trình ETL
│   └── warehouse/
│       ├── instacart_dw.duckdb  # Kho Data Warehouse hợp nhất
│       └── marts/           # Dữ liệu xuất dùng cho Modeling
├── src/                     # Source Code xử lý
│   ├── etl/                 # Các script trích xuất - biến đổi - nạp
│   └── marts/               # Các script trích xuất dữ liệu thành Marts
├── notebook/                # Phòng Lab (Jupyter): Khám phá EDA, vẽ biểu đồ rác và test nháp thuật toán Model
├── evaluation/              # Thư mục chứa báo cáo kết quả đánh giá (benchmark_results.txt)
├── report/                  # Thư mục chứa Báo cáo phân tích tĩnh (.md)
├── app.py                   # Triển khai Streamlit Dashboard
├── run_etl.py               # Chạy quy trình ETL và xuất Data Marts
├── evaluate_models.py       # Kịch bản Evaluation mô hình AI trên Terminal
├── clean.py                 # Kịch bản xóa toàn bộ Cache, DB, khôi phục Gốc
├── .env                     # File cấu hình biến số môi trường (Tỉ lệ Sampling)
└── requirements.txt         # Danh sách thư viện cần thiết
```

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

- **Cơ sở dữ liệu (OLAP)**: DuckDB
- **Xử lý Dữ liệu**: Pandas, PyArrow (Parquet Format)
- **Khai Phá Dữ liệu (ML/DM)**: MLXTend (Apriori/FPGrowth), PyTorch Geometric (LightGCN/GNN)
- **Giao diện & Đồ thị trực quan**: Streamlit, Plotly Express, Graphviz

---

## 💻 Hướng Dẫn Cài Đặt Và Sử Dụng (How to run)

### 1. Thiết lập Môi trường (Virtual Environment)
Yêu cầu đã cài đặt `Python 3.9+`. Tạo và kích hoạt môi trường `.venv`:
```powershell
# Chạy trên PowerShell của Windows
python -m venv .venv
.\.venv\Scripts\activate
```

Cài đặt các thư viện cần thiết:
```powershell
pip install -r requirements.txt
```

### 1.5. Nạp Dữ liệu Gốc (Dành cho Người Mới / Giảng Viên)
Do nền tảng GitHub từ chối chứa tệp tin quá lớn (vượt giới hạn 100MB), Dữ liệu Gốc (`Raw Data`) đã bị chặn và không có trên Repository trực tuyến.
Để chạy được mã nguồn nguyên vẹn trên máy giảng viên hoặc cộng sự mới, xin vui lòng:
1. Bạn hãy giải nén nguồn Data Gốc từ file ZIP do trưởng nhóm cung cấp (tên là `raws.zip` hoặc tương tự).
2. Lấy toàn bộ các file `*.csv` (ví dụ: `orders.csv`, `order_products__prior.csv`...).
3. Chép đè toàn bộ chúng vào đúng đường dẫn bảo tồn: `data/raws/`.
*(Chú ý: Hãy kiểm tra chắc chắn tệp `data/raws/orders.csv` đã hiện diện với dung lượng ~100MB, và file prior là ~550MB trước khi đi tiếp).*

### 2. Xây Dựng Kho Dữ Liệu (Run ETL Pipeline)
Sau khi đảm bảo các file CSV thô đã nằm an toàn trong thư mục `data/raws/`, hãy khởi chạy quá trình nhào nặn Database lớn nhất dự án (bước này sẽ mất vài phút tuỳ sức cày của CPU do phải xử lý vòng lặp 32 triệu Transaction):

```powershell
python run_etl.py
```
*Hệ thống sẽ chạy qua 3 bước: Extract -> Transform -> Load và kết thúc bằng việc sinh ra Data Marts.*

### 3. Trải nghiệm Giao diện UI (Streamlit Dashboard)
Mở tính năng vẽ Biểu Đồ Truy Vấn Thời Gian Thực và thử nghiệm Suggestion Món Hàng bằng cách chạy lệnh:

```powershell
streamlit run app.py
```
*Trình duyệt sẽ tự động mở lên tại địa chỉ `http://localhost:8501`. Tất cả tinh tuý đồ hoạ của dự án trình bàny đều nằm ở đây.*

### 4. Đào Sâu Tri Thức Mở Rộng & Đánh Giá Tự Động (Benchmark)
Để kiểm chứng và so sánh sức mạnh thực sự của Mô hình Học Sâu (LightGCN) so với Luật Kết Hợp truyền thống (FP-Growth), hãy chạy kịch bản Benchmark:
```powershell
python evaluate_models.py
```
*Lưu ý: Mặc định script tự động lấy 20% lượng Data Warehouse (cấu hình qua biến số `SAMPLE_DATA_RATIO` tại file `.env` root). Kết quả thi (Recall, Precision, F1-Score) sẽ tự động xuất vào `evaluation/metrics.json` và `evaluation/benchmark_results.txt` để vẽ lên Streamlit Dashboard.*

### 5. Dọn dẹp Dự án (Tùy chọn)
Nếu bạn muốn Reset dự án từ đầu (hoặc trước khi nén gửi GV), hãy chạy lệnh:
```powershell
python clean.py
```
*Lệnh này an toàn xóa toàn bộ tệp sinh ra (Parquet, DuckDB, File Data Marts, Code Cache), chỉ giữ lại mã nguồn và file Kaggle nguyên thủy.*

---
💡 **Ghi chú Dành cho Giảng Viên chấm điểm:** 
Hệ thống này được mô phỏng theo **Quy trình Phần Mềm Dữ Liệu Thực Tế**. Tức là:
- Thư mục **`notebook/`** đóng vai trò là Khu vực Làm nháp (R&D). Chứa các file `.ipynb` ghi nhận quá trình sinh viên tự vẽ biểu đồ thăm dò (EDA), xử lý Data thô sơ cấp và test nháp các thuật toán Machine Learning. 
- Mọi tư duy tinh tuý trong `notebook/` sau đó đã được **chuẩn hoá, đóng gói và tái cấu trúc (Refactor)** thành luồng mã code tự động nằm ở các tệp `.py` (gồm `run_etl.py` để xử lý ETL và `evaluate_models.py` phục vụ chạy thuật toán tự động trên Terminal). Điều này giúp Đồ án bứt phá khỏi dạng "Bài tập thực hành Notebook" để tiến tới một "Application Dữ liệu".
