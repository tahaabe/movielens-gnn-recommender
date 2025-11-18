import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class SAGERecommender(nn.Module):
    def __init__(self, in_feats=1, hidden_size=16, out_feats=1):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size)
        self.conv2 = SAGEConv(hidden_size, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


@st.cache_resource
def load_data_and_model():

    
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    df = pd.read_csv(url, sep="\t", names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Charger le fichier u.item (titres des films)
    movies_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    movies_cols = ['movieId', 'title', 'release_date', 'video_release', 'imdb_url'] + \
        ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv(movies_url, sep="|", names=movies_cols, encoding="latin-1")

    # Nombre users / items
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()

    # Mapping pour PyTorch Geometric
    user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    item_map = {iid: i + n_users for i, iid in enumerate(df['item_id'].unique())}

    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)

    # Edge index
    edge_index = torch.tensor([
        df['user_idx'].values,
        df['item_idx'].values
    ], dtype=torch.long)

    # Labels = ratings
    ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    # Features = vecteur [1]
    node_features = torch.ones(n_users + n_items, 1)

    # Construire le graphe
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=ratings,
        num_nodes=n_users + n_items
    )

 
    model = SAGERecommender()
    model.load_state_dict(torch.load("gnn_recommender.pt", map_location="cpu"))
    model.eval()

    return model, data, user_map, item_map, movies


model, data, user_map, item_map, movies = load_data_and_model()



# Fonction de recommandation GNN


def recommend_gnn(user_id, top_n=10):
    if user_id not in user_map:
        return []

    with torch.no_grad():
        # embeddings for all nodes
        emb = model(data)[:, 0]          # shape: [num_nodes]

        # user embedding
        user_node = user_map[user_id]
        user_emb = emb[user_node]

        # item nodes + embeddings
        item_nodes = torch.tensor(list(item_map.values()))
        item_embs = emb[item_nodes]

        # scores = produit scalaire
        scores = (user_emb * item_embs).cpu().numpy()

        # top-k indices (numpy)
        import numpy as np
        top_idx = np.argsort(scores)[-top_n:][::-1]
        top_idx = top_idx.copy()  

       
        top_item_nodes = item_nodes[top_idx.tolist()].cpu().numpy()
        top_scores = scores[top_idx]

      
        node_to_item = {v: k for k, v in item_map.items()}
        top_movie_ids = [node_to_item[int(n)] for n in top_item_nodes]

        return list(zip(top_movie_ids, top_scores.tolist()))



# Interface Streamlit


st.title("🎬 Recommandation de Films (GNN GraphSAGE)")
st.write("Modèle entraîné sur MovieLens 100k")

# Sélecteur d’utilisateur
all_user_ids = sorted(user_map.keys())
selected_user = st.selectbox("Sélectionne un utilisateur :", all_user_ids)

k = st.slider("Nombre de recommandations (Top-K)", 3, 20, 10)

if st.button("Générer les recommandations"):
    recs = recommend_gnn(selected_user, top_n=k)

    st.subheader(f"Top {k} recommandations pour l’utilisateur {selected_user}")

    table_rows = []
    for movie_id, score in recs:
        title = movies[movies['movieId'] == movie_id]['title'].values
        title = title[0] if len(title) > 0 else "Unknown"

        table_rows.append({
            "movieId": movie_id,
            "title": title,
            "score": float(score)
        })

    st.table(pd.DataFrame(table_rows))
