import torch
import time
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import LightGCN

def prepare_graph_data(train_df, all_products):
    """
    Chuẩn bị Bipartite Graph
    """
    order_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    product_encoder.fit(all_products)
    train_df['order_idx'] = order_encoder.fit_transform(train_df['order_id'])
    train_df['product_idx'] = product_encoder.transform(train_df['product_name'])

    num_orders = len(order_encoder.classes_)
    num_products = len(product_encoder.classes_)
    num_nodes = num_orders + num_products

    train_df['item_node_idx'] = train_df['product_idx'] + num_orders
    edge_1 = torch.tensor([train_df['order_idx'].values, train_df['item_node_idx'].values], dtype=torch.long)
    edge_2 = torch.tensor([train_df['item_node_idx'].values, train_df['order_idx'].values], dtype=torch.long)
    edge_index = torch.cat([edge_1, edge_2], dim=1)
    
    metadata = {
        'num_orders': num_orders,
        'num_products': num_products,
        'num_nodes': num_nodes,
        'product_encoder': product_encoder,
        'order_encoder': order_encoder,
        'edge_1': edge_1
    }
    return edge_index, metadata

def train_lightgcn(edge_index, metadata, epochs=30, dim=64):
    """
    Huấn luyện mạng LightGCN
    """
    num_nodes = metadata['num_nodes']
    num_orders = metadata['num_orders']
    edge_1 = metadata['edge_1']
    
    model = LightGCN(num_nodes=num_nodes, embedding_dim=dim, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out_emb = model.get_embedding(edge_index)
        pos_edge = edge_1
        neg_items = torch.randint(num_orders, num_nodes, (pos_edge.size(1),))
        neg_edge = torch.stack([pos_edge[0], neg_items], dim=0)
        
        pos_scores = (out_emb[pos_edge[0]] * out_emb[pos_edge[1]]).sum(dim=-1)
        neg_scores = (out_emb[pos_edge[0]] * out_emb[neg_edge[1]]).sum(dim=-1)
        
        loss = model.recommendation_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        
    gnn_time = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        item_emb = model.get_embedding(edge_index)[num_orders:]
        
    return model, item_emb, gnn_time

def predict_gnn_bipartite(input_basket, item_emb, metadata, k=10):
    """
    Dự đoán (Inference) dựa vào Dot Product của trung bình rổ hàng
    """
    product_encoder = metadata['product_encoder']
    
    try:
        input_idx = product_encoder.transform(input_basket)
    except: return []
    if len(input_idx) == 0: return []
    
    basket_vector = item_emb[input_idx].mean(dim=0)
    scores = torch.matmul(basket_vector, item_emb.T)
    scores[input_idx] = -float('inf')
    
    top_indices = scores.topk(k).indices.numpy()
    return product_encoder.inverse_transform(top_indices).tolist()
