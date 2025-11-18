import torch
from torch_geometric.data import Data

def build_graph(df):
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()

    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: i + n_users for i, iid in enumerate(df['item_id'].unique())}

    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)

    edge_index = torch.tensor(
        [df['user_idx'].values, df['item_idx'].values],
        dtype=torch.long
    )

    ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    node_features = torch.ones(n_users + n_items, 1)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=ratings,
        num_nodes=n_users + n_items
    )

    return data, user_map, item_map
