import numpy as np
import pandas as pd

def create_holdout_test_set(df, n_test_orders=500, seed=42):
    """
    Tạo tập Test bằng cách:
    1. Chỉ chọn các đơn hàng có từ 3 sản phẩm trở lên
    2. Tách mỗi giỏ hàng làm 2 phần: Khách đang chọn (Input) và Khách sẽ mua (Target)
    """
    order_sizes = df.groupby('order_id').size()
    valid_orders = order_sizes[order_sizes >= 3].index.tolist()

    np.random.seed(seed)
    test_order_ids = set(np.random.choice(valid_orders, size=n_test_orders, replace=False))

    train_df = df[~df['order_id'].isin(test_order_ids)].copy()
    test_df = df[df['order_id'].isin(test_order_ids)].copy()

    test_cases = []
    for order_id, group in test_df.groupby('order_id'):
        items = group['product_name'].tolist()
        np.random.shuffle(items)
        
        # Chia đôi giỏ hàng: 50% làm input, 50% còn lại làm target
        split_idx = len(items) // 2
        input_items = items[:split_idx]
        target_items = items[split_idx:]
        
        test_cases.append({
            'order_id': order_id,
            'input': input_items,
            'target': target_items
        })
            
    return train_df, test_cases


def calculate_hit_rate(test_cases, model_type, model_predictor, **kwargs):
    """
    Chấm điểm dựa trên "Tổng số món đồ dự đoán đúng" (True Positives)
    """
    k = kwargs.get('k', 10)
    
    total_correct_items = 0    # Tổng đồ đoán trúng
    total_target_items = 0     # Tổng đồ khách thực sự mua rải rác trong tập Test
    total_predicted_items = 0  # Tổng đồ hệ thống đã tư vấn
    
    for case in test_cases:
        targ = set(case['target'])
        input_items = case['input']
        
        preds = set(model_predictor(input_items, **kwargs))
        
        # Đếm số món đoán trúng trong đơn này
        hits = len(preds.intersection(targ))
        
        total_correct_items += hits
        total_target_items += len(targ)
        total_predicted_items += len(preds)
            
    # Tính toán Precision và Recall
    precision = total_correct_items / total_predicted_items if total_predicted_items > 0 else 0
    recall = total_correct_items / total_target_items if total_target_items > 0 else 0
    
    return total_correct_items, total_target_items, precision, recall


def predict_apriori(input_basket, rules, k=10):
    """
    Dự đoán sản phẩm bằng luật kết hợp (Apriori/FP-Growth)
    """
    basket_set = set(input_basket)
    recs = rules[rules['antecedents'].apply(lambda x: x.issubset(basket_set))]
    
    suggested = []
    for item_set in recs['consequents']:
        for item in item_set:
            if item not in basket_set and item not in suggested:
                suggested.append(item)
                if len(suggested) == k:
                    return suggested
    return suggested
