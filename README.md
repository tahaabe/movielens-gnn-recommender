# 🎬 MovieLens GNN Recommender System

Ce projet implémente un système de recommandation de films basé sur un **Graph Neural Network (GNN)** utilisant **GraphSAGE**, entraîné sur le dataset **MovieLens 100k**.  
L'application finale est déployée avec **Streamlit**.

---

## 📌 1. Description du projet

L’objectif est de prédire quels films un utilisateur pourrait aimer, en utilisant :

- un graphe biparti **utilisateur → film**
- un modèle GNN (**SAGEConv**)
- un mécanisme de recommandation basé sur les embeddings appris
- une interface utilisateur simple développée avec Streamlit

Le projet suit un pipeline machine learning  :
1. Préparation des données
2. Construction du graphe PyTorch Geometric
3. Définition du modèle GNN
4. Entraînement
5. Génération de recommandations
6. Déploiement Streamlit

---

## 📦 2. Dataset : MovieLens 100k

Nous utilisons le dataset public MovieLens :
https://grouplens.org/datasets/movielens/100k/

Il contient :
- 100 000 notes
- 943 utilisateurs
- 1682 films

---

## 🏗️ 3. Structure du projet

movielens-gnn-recommender/
│
├── app.py # Application Streamlit
├── requirements.txt 
├── README.md # 
├── .gitignore
│
├── model/
│ └── gnn_recommender.pt # Modèle entraîné
│
└── src/
├── model.py # Réseau GNN (GraphSAGE)
├── build_graph.py 
├── data_preparation.py 
├── recommend.py 



## ▶️ 4. Installation

Assurez-vous d’avoir **Python 3.11**.

### 1) Cloner le projet

git clone https://github.com/<votre-username>/movielens-gnn-recommender.git
cd movielens-gnn-recommender
2) Installer les dépendances
bash
Copier le code
pip install -r requirements.txt
🚀 5. Lancer l'application Streamlit
Assurez-vous que le modèle gnn_recommender.pt se trouve dans /model.

Puis lancez :

bash
Copier le code
streamlit run app.py
L’interface s’ouvrira dans votre navigateur :
http://localhost:8501

6. Modèle GraphSAGE
Le modèle utilise :

une couche GraphSAGE de 16 dimensions

une activation ReLU

une couche de sortie GraphSAGE

un score prédictif basé sur le produit des embeddings utilisateur/film

7. Démonstration
L’utilisateur sélectionne un ID utilisateur puis reçoit les Top-K films recommandés avec leurs scores.

8. Auteur
Projet réalisé par TAHA EL BEKKALI dans le cadre d’un projet end to end ML (MovieLens 100k + GNN)