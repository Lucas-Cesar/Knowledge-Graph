import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# imports usados para divisão de grupos
from node2vec import Node2Vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

pd.set_option('display.max_columns', None)
df = pd.read_csv("vgsales.csv")

n_jogos = 35
df = df.head(n_jogos)


G = nx.Graph()
G.add_nodes_from(df['Name'])

for _, row in df.iterrows():
    G.add_edge(row['Name'], row['Platform'], label='')
    G.add_edge(row['Name'], int(row['Year']), label='')
    G.add_edge(row['Name'], row['Genre'], label='')
    G.add_edge(row['Name'], row['Publisher'], label='')
    G.add_edge(row['Name'], row['classe_vendas'],
               label='')  # relação por vendas(quantidade de vendas na mesma categoria)

relacoes = {}

for _, row in df.iterrows():
    G.add_edge(row['Name'], row['Platform'], label='')
    relacoes[(row['Name'], row['Platform'])] = 'Plataforma'

    G.add_edge(row['Name'], int(row['Year']), label='')
    relacoes[(row['Name'], int(row['Year']))] = 'Ano'

    G.add_edge(row['Name'], row['Genre'], label='')
    relacoes[(row['Name'], row['Genre'])] = 'Gênero'

    G.add_edge(row['Name'], row['Publisher'], label='')
    relacoes[(row['Name'], row['Publisher'])] = 'Publicadora'

    G.add_edge(row['Name'], row['classe_vendas'], label='')
    relacoes[(row['Name'], row['classe_vendas'])] = 'Classe de Vendas'


def numero_de_vertices():
    return G.number_of_nodes()


def numero_de_arestas():
    return G.number_of_edges()


def grau(v):
    degree_centrality = nx.degree_centrality(G)
    return degree_centrality.get(v, None)


def verticesAdjacentes(v):
    if v in G:
        return list(G.neighbors(v))
    else:
        print(f"O vértice '{v}' não está presente no grafo.")
        return []


def arestasAdjacentes(v):
    if v in G:
        return list(G.edges(v))
    else:
        print(f"O vértice '{v}' não está presente no grafo.")
        return []


def saoAdjacentes(v1, v2):
    if v1 in G and v2 in G:
        return G.has_edge(v1, v2)
    else:
        print(f"Pelo menos um dos vértices fornecidos não está presente no grafo.")
        return False


def insereVertice(v):
    if v not in G:
        G.add_node(v)
        print(f"Vértice '{v}' inserido com sucesso.")
    else:
        print(f"O vértice '{v}' já está presente no grafo.")


def insereAresta(v1, v2, relacao):
    if v1 in G and v2 in G:
        if not G.has_edge(v1, v2):
            G.add_edge(v1, v2, label=relacao)
            relacoes[(v1, v2)] = relacao
            print(f"Aresta entre '{v1}' e '{v2}' com relação '{relacao}' inserida com sucesso.")
        else:
            print(f"Aresta entre '{v1}' e '{v2}' já está presente no grafo.")
    else:
        print(f"Pelo menos um dos vértices fornecidos não está presente no grafo.")


def removeVertice(v):
    if v in G:
        G.remove_node(v)
        print(f"Vértice '{v}' removido com sucesso.")
    else:
        print(f"O vértice '{v}' não está presente no grafo.")


def removeAresta(v1, v2):
    if v1 in G and v2 in G:
        if G.has_edge(v1, v2):
            G.remove_edge(v1, v2)
            print(f"Aresta entre '{v1}' e '{v2}' removida com sucesso.")
        else:
            print(f"Aresta entre '{v1}' e '{v2}' não está presente no grafo.")
    else:
        print(f"Pelo menos um dos vértices fornecidos não está presente no grafo.")


def obter_relacao(v1, v2):
    if (v1, v2) in relacoes:
        return relacoes[(v1, v2)]
    elif (v2, v1) in relacoes:
        return relacoes[(v2, v1)]
    else:
        return None


print(numero_de_arestas())
# Visualize the shortest path
pos = nx.spring_layout(G, seed=80, k=1.4)
labels = nx.get_edge_attributes(G, 'label')


#Plotar o grafo sem a divisão de grupos
plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, font_size=8,font_color='black',font_weight='bold', node_size=400, node_color='gray', edge_color='gray', alpha=0.6)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6, font_color='gray', label_pos=0.3, verticalalignment='baseline')
plt.title('Knowledge Graph')
plt.show()


# Divisão em grupos:
edge_labels = {(source, target): data['label'] for source, target, data in G.edges(data=True)}

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # You can adjust these parameters
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Training the model
embeddings = np.array([model.wv[str(node)] for node in G.nodes()])

# Dividir em grupos usando clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, font_size=10, node_size=400, node_color=cluster_labels, cmap=plt.cm.Set1,
        edge_color='gray', alpha=0.6)
plt.title('Graph Clustering using K-Means')

plt.show()




