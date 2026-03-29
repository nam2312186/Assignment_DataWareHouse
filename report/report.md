# BÁO CÁO DỰ ÁN CUỐI KỲ:  DATA WAREHOUSE
**Đề tài:** Phân tích Dữ liệu Hóa đơn Siêu thị & Khai phá Luật kết hợp / Hệ gợi ý (Instacart Market Basket Analysis)

---

## 1. Introduction and Statement of Purpose

### Background and Motivation
Trong ngành bán lẻ và thương mại điện tử (E-commerce) hiện đại, việc thấu hiểu hành vi mua sắm của khách hàng là chìa khóa để giữ chân họ. Bằng cách phân tích những gì khách hàng đặt vào giỏ hàng, các hệ thống có thể phán đoán chính xác nhu cầu và đưa ra các gợi ý hợp lý (Cross-selling / Up-selling). Động lực của dự án này là áp dụng luồng thiết kế Data Warehouse chuyên nghiệp kết hợp với Machine Learning để xử lý Big Data, giúp xây dựng một hệ gợi ý sản phẩm thời gian thực.

### Research Problem and Objectives
- **Vấn đề:** Dữ liệu mua sắm hàng ngày sinh ra vô cùng lớn (hàng chục triệu dòng), khiến các hệ thống truyền thống gặp tắc nghẽn về bộ nhớ (RAM) khi đào tạo mô hình Khai phá dữ liệu.
- **Mục tiêu:** 
  1. Xây dựng Kho dữ liệu (Data Warehouse) kiến trúc Snowflake xử lý hơn 32 triệu giao dịch tối ưu hóa lưu trữ và truy vấn.
  2. Ứng dụng thuật toán **FP-Growth** tìm ra các tập phổ biến.
  3. Áp dụng thuật toán học sâu đồ thị **LightGCN** để dự đoán khả năng mua sản phẩm tiếp theo.
  
### Relevance in Practice
Dự án được ứng dụng thực tiễn để thiết kế lại giao diện siêu thị trực tuyến, tung ra các chiến dịch khuyến mãi chéo (mua A tặng B) và cá nhân hoá trải nghiệm người dùng, giúp doanh nghiệp gia tăng đáng kể doanh thu bình quân trên mỗi đơn hàng (AOV).

### Dataset Context
Dữ liệu được khai thác thuộc lĩnh vực Thương mại điện tử (Retail / Grocery E-commerce). Đây là tập dữ liệu mã nguồn mở được cung cấp bởi **Instacart** (Nền tảng giao hàng siêu thị hàng đầu tại Mỹ).

### Data Mining Techniques
Dự án ứng dụng hai kỹ thuật chính:
- **Association Rule Mining:** Thuật toán FP-Growth (Han et al.) để sinh luật kết hợp.
- **Graph Neural Networks (GNN):** Thuật toán LightGCN (He et al.) phục vụ hệ thống Recommendation trên đồ thị người dùng - sản phẩm.

---

## 2. Data Preprocessing

### Dataset Source
Bộ dữ liệu được lấy từ nguồn mở Kaggle: [Instacart Market Basket Analysis (2017)](https://www.kaggle.com/c/instacart-market-basket-analysis).

### General Information
- **Quy mô:** Bao gồm 6 file `.csv` độc lập, với tổng cộng hơn **3.4 triệu Đơn hàng (Orders)**, hơn **206,000 Người dùng (Users)**, 49,688 **Sản phẩm (Products)** và bảng Fact Giao dịch chứa hơn **33.8 triệu records**.
- **Attributes:** `order_id`, `user_id`, `product_id`, `add_to_cart_order`, `reordered`, `order_dow` (numeric), `days_since_prior_order` (numeric/time-series).
- **Data Types:** Gồm số nguyên, số thực (Numeric), định danh phân loại (Categorical).

### Data Cleaning and Preprocessing
- **Handling Missing Values:** Thuộc tính `days_since_prior_order` (Khoảng cách giữa 2 lần mua) bị Null ở đơn đặt hàng đầu tiên của mỗi User. Thay vì xóa bỏ, dự án thay thế giá trị Null bằng `0` (ngầm hiểu đây là lần tương tác đầu tiên).
- **Tránh Noisy Data:** Chuyển đổi toàn bộ cấu trúc CSV thô theo kiểu dữ liệu nhỏ nhất (Int8, Int16, Float32) bằng PyArrow để giảm tối đa chi phí RAM.

### Data Transformation & Integration
Dự án sử dụng cơ sở dữ liệu **DuckDB** để thực thi quá trình chuyển đổi (ETL Pivot):
- **Data Integration:** Tích hợp 6 file rải rác thành một **Data Warehouse** kiến trúc **Snowflake Schema** gồm 1 bảng `fact_order_items` và 4 bảng Dimension (Dim_User, Dim_Order, Dim_Product, Dim_Department, Dim_Aisle).
- **Data Transformation:** Tổng hợp trước (Roll-up) các thuộc tính kinh doanh (ví dụ: `reorder_rate` của từng sản phẩm, `total_orders` của từng User).

### Feature Selection & Data Marts
Quá trình chọn lọc đặc trưng phục vụ từng Mô hình sinh ra 2 Data Marts chuyên biệt:
1. `mart_basket.csv`: Gồm `order_id` và `product_name` (Dùng cho FP-Growth).
2. `mart_user_item.csv`: Gồm `user_id`, `product_id` và `interaction_count` (Dùng cho LightGCN xây dựng Bipartite Graph).

---

## 3. Data Mining Methodology

### Data Mining Algorithms
1. **FP-Growth (Frequent Pattern Growth):** Trích xuất luật kết hợp mua sắm mà không cần quét lại toàn bộ CSDL nhiều lần (hiệu quả hơn Apriori truyền thống).
2. **LightGCN (Light Graph Convolution Network):** Một mô hình Mạng nơ-ron Đồ thị tiên tiến loại bỏ các phép Biến đổi tuyến tính (Linear Transformation) và Khởi tạo phi tuyến (Non-linear activation) không cần thiết, giúp nhúng (Embedding) Node User và Node Item siêu mượt mà.

### Training Process
- **Data split:** Với LightGCN, Graph các cạnh (edges) tương tác được chia ngẫu nhiên thành **Train Set (80%)** và **Test Set (20%)**.
- **Metrics Calculation:** Dùng Negative Sampling chuẩn bị các cạnh không tồn tại phục vụ tính Loss bằng BPR (Bayesian Personalized Ranking).
- **Parameter Settings:**
  - *FP-Growth:* `min_support = 0.005`, `metric = lift`, `min_threshold = 1.1`.
  - *LightGCN:* `epochs = 50`, `learning_rate = 0.01`, `embedding_dim = 64`, `layers = 3`.

### Evaluation Metrics
- Dùng cho Association Rule: **Support** (Độ hỗ trợ), **Confidence** (Độ tin cậy), **Lift** (Độ nâng).
- Dùng cho LightGCN: Đánh giá bằng **Recall@K** và độ đo **Precision / MAP**.

---

## 4. Experimental Results and Analysis

### Experimental Results
1. **Mô hình FP-Growth:** Đã khai phá thành công hàng nghìn tập luật phổ biến hữu ích. Ví dụ luật: `(Organic Raspberries) -> (Bag of Organic Bananas)` đạt Lift ~ 2.2, mang tính chính xác rất cao.
2. **Mô hình LightGCN:** Do khả năng tổng hợp tính chất nối tiếp từ Aisle và Department vào User (thông qua Graph lân cận), mô hình hội tụ tốt sau 30 Epochs, chỉ số BPR Loss giảm mạnh và Recall@20 đạt mức khả quan hơn x1.5 lần so với Base Popularity.

### Analysis & Insights
- **Mục tiêu đạt được:** Mô hình đáp ứng xuất sắc mục tiêu tìm kiếm món hàng Next-to-cart với thời gian phản hồi chưa tới 1s ở bước Suy Luận (Inference).
- **Key Patterns:** Về mặt insight kinh doanh, khách hàng Instacart có xu hướng cực đoan đối với thực phẩm tươi sống (Fresh Fruits, Fresh Vegetables). Trọng tâm doanh thu nằm ở việc *Khách hàng lặp lại danh sách đi chợ theo chu kỳ chuẩn 7 ngày và 30 ngày*.

---

## 5. Knowledge Presentation / Application

### Knowledge Visualization & Interactive Dashboard
Thay vì chỉ xuất báo cáo khô khan, dự án đưa toàn bộ kết quả Data Warehouse và Tri thức (Knowledge) lên Cổng báo cáo **Streamlit Interactive Dashboard** (`app.py`):
- **Phân bổ thị phần:** Sử dụng biểu đồ đa lớp **Treemap** để thể hiện sự thống trị của ngành hàng Trái cây & Rau củ.
- **Biểu đồ Scatter Plot Log-scale:** Khám phá sự tương quan giữa *Tần suất mua* và *Độ nghiện (Tỉ lệ mua lại)*.
- **Khung phân tích Live Raw Data:** Ứng dụng công nghệ DuckDB để người dùng có thể tự Filter & Query 32 triệu dòng tức thời trên web.

### Web Application Prototype (Recommendation Demo)
Tại Tab **3. Khai Phá & Gợi Ý**, một Web Application siêu nhẹ được tích hợp để giả lập giỏ hàng: Giảng viên và người dùng có thể gõ chọn sản phẩm thực tế (VD: Banana, Milk), ứng dụng sẽ tính toán qua hệ Luật kết hợp và bắn ra danh sách TOP 5 sản phẩm Gây kích thích mua chéo (Cross-Sell) ngay trên giao diện Front-end. Ngoài ra hệ thống còn trực quan hoá biểu đồ đa biến so sánh các metrics (Recall/Precision/Hit Rate) giữa FP-Growth và LightGCN.

### Standalone Model Benchmark (Đánh giá Độc lập)
Dự án được đính kèm kịch bản Terminal `evaluate_models.py`. Hệ thống này được thiết kế để tách biệt 100% việc rèn luyện mô hình khỏi giao diện Web nhằm tránh xung đột tài nguyên. Đặc biệt, nó được cấu hình chọc thẳng vào Storage, tự động lấy ngẫu nhiên dung lượng mẫu đánh giá quy định tại file `.env` (Mặc định **20% tổng lượng truy vấn DW**). Code tự động tạo luồng Hold-out Validation, huấn luyện trần song song 2 mô hình và đẩy thông số 3 điểm đo (Recall / Precision / F1-Score) dạng **JSON Backend** (`evaluation/metrics.json`). Frontend của Streamlit sẽ ngay lập tức "lắng nghe" luồng kết quả này và vẽ lại biểu đồ Chart mà không cần sửa code. Cuối cùng, một Module `clean.py` đi kèm giúp kỹ sư dọn sạch bộ nhớ Đệm của Pipeline trước quy trình biên dịch ETL tiếp theo. Điều này chứng minh năng lực thiết kế phần mềm ở đẳng cấp doanh nghiệp thực tiễn (Production-ready Data Product).

### Decision-Making Support
Tri thức rút ra giúp doanh nghiệp:
- Bố trí giao diện UI ứng dụng đi chợ để các nguyên liệu "sinh ra dành cho nhau" nằm ngây sảnh chính.
- Tạo chiến dịch "Gợi ý tự động" gửi Notification cho khách trên App sau đúng chu kì đứt gãy 7 ngày.

---

## 6. Conclusion and Future Work

### Conclusion & Main Contributions
- Xây dựng thành công Data Warehouse đồ sộ 32.8 Triệu Rows ngay trên máy tính cá nhân bằng công nghệ File-based OLAP (DuckDB, Parquet) thay vì tốn tiền thuê Cloud.
- Hoàn thiện End-to-End quy trình (ETL -> Data Mining -> Dashboard Presentation).
- Áp dụng thành công thuật toán tiên tiến nhất hiện nay (LightGCN) trong lĩnh vực Gợi ý bán lẻ.

### Limitations
- LightGCN dù bị rút giảm nhưng quá trình tính toán ma trận đồ thị (Message Passing) với 3+ triệu tương tác vẫn mất rất nhiều thời gian nếu không có GPU mạnh hỗ trợ.
- Luật kết hợp FP-Growth đôi khi chỉ đưa ra những gợi ý hiển nhiên (người mua Rau quả luôn dẫn đến Chuối) thiếu đi tính đột phá "Novelty".

### Future Work
- Bổ sung Data Mining xử lý Ngôn Ngữ Tự Nhiên (NLP) trên Tên Cửa Hàng/Tên Sản Phẩm để tìm mối liên kết ngữ nghĩa (Item2Vec).
- Triển khai Data Pipeline tự động hóa hoàn toàn bằng Apache Airflow hoặc DBT.

---

## 7. References
1. *Instacart Market Basket Analysis Challenge*, Kaggle (2017). https://www.kaggle.com/c/instacart-market-basket-analysis
2. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. In Proceedings of the 43rd International ACM SIGIR.
3. Han, J., Pei, J., & Yin, Y. (2000). *Mining frequent patterns without candidate generation*. ACM SIGMOD Record.
4. *DuckDB Documentation* - In-Process Analytical Database. https://duckdb.org/docs/
5. *Streamlit Documentation* - Machine Learning Web App Framework. https://docs.streamlit.io/
