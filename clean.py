import os
import shutil
from pathlib import Path

def clean_directory(dir_path: Path):
    if dir_path.exists() and dir_path.is_dir():
        try:
            shutil.rmtree(dir_path)
            print(f"✅ Đã xóa thư mục: {dir_path}")
        except Exception as e:
            print(f"❌ Không thể xóa {dir_path}: {e}")

def clean_file(file_path: Path):
    if file_path.exists() and file_path.is_file():
        try:
            os.remove(file_path)
            print(f"✅ Đã xóa file: {file_path}")
        except Exception as e:
            print(f"❌ Không thể xóa {file_path}: {e}")

def main():
    print("="*60)
    print("🧹 INSTACART DATA WAREHOUSE - CÔNG CỤ LÀM SẠCH (CLEANER)")
    print("="*60)
    print("Chương trình sẽ xóa các dữ liệu được sinh ra trong quá trình chạy ETL/Model.")
    print("Mục đích: Khôi phục dự án về trạng thái ban đầu sạch sẽ để Demo từ A-Z.\n")
    
    # Danh sách các thư mục tạm / dữ liệu sinh ra cần xóa
    directories_to_remove = [
        Path("data/staging"),             # File Parquet trung gian
        Path("data/warehouse/marts"),     # File Data Mart xuất ra
        Path("evaluation"),               # File JSON và TXT kết quả
    ]
    
    # Danh sách các file cốt lõi cần làm sạch
    files_to_remove = [
        Path("data/warehouse/instacart_dw.duckdb"),         # File cơ sở dữ liệu DuckDB
        Path("evaluation/artifacts/lightgcn_checkpoint.pt"), # Model checkpoint LightGCN
    ]
    
    print("[1] Đang dọn file CSDL DuckDB...")
    for f in files_to_remove:
        clean_file(f)

    print("\n[2] Đang dọn thư mục Data Marts, Staging và Kết quả...")
    for d in directories_to_remove:
        clean_directory(d)
        
    print("\n[3] Đang dọn dẹp các tệp rác hệ thống (__pycache__)...")
    for p in Path('.').rglob('__pycache__'):
        clean_directory(p)
        
    print("\n🎉 KHÔI PHỤC HOÀN TẤT! Dự án đã rỗng và sạch sẽ chuyên nghiệp.")
    print("Bạn có thể bật lại luồng theo thứ tự sau:")
    print("   👉 Bước 1: python run_etl.py")
    print("   👉 Bước 2: python evaluate_models.py")
    print("   👉 Bước 3: streamlit run app.py\n")

if __name__ == "__main__":
    main()
