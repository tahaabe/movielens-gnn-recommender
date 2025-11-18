import torch
import numpy as np

def recommend_gnn(model, data, user_map, item_map, movies, user_id, top_n=10):
    if user_id not in user_map:
        return None

    with torch.no_grad():
        emb = model(data)[:, 0]

        user_node = user_map[user_id]
        user_emb = emb[user_node]

        item_nodes = torch.tensor(list(item_map.values()))
        item_embs = emb[item_nodes]

        scores = (user_emb * item_embs).cpu().numpy()

        top_idx = np.argsort(scores)[-top_n:][::-1]
        top_idx = top_idx.copy()

        top_item_nodes = item_nodes[top_idx.tolist()].cpu().numpy()
        top_scores = scores[top_idx]

        node_to_item = {v: k for k, v in item_map.items()}
        top_movie_ids = [node_to_item[int(n)] for n in top_item_nodes]

        # Ajouter les titres
        results = []
        for movie_id, score in zip(top_movie_ids, top_scores):
            title = movies[movies['movieId'] == movie_id]['title'].values
            title = title[0] if len(title) else "Unknown"
            results.append((movie_id, title, float(score)))

        return results
