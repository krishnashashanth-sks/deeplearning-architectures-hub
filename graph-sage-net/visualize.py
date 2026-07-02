from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
import networkx as nx

def visualize_graph(data, title="Karate Club Graph"):
    plt.figure(figsize=(8, 8))
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout
    colors = data.y.tolist() # Use node labels for coloring
    nx.draw_networkx(G, pos, cmap=plt.cm.get_cmap('viridis', max(colors) + 1), node_color=colors, node_size=300, with_labels=True)
    plt.title(title)
    plt.show()
    
def visualize_embeddings_sage(data, model):
    model.eval()
    with torch.no_grad():
        # We need to get the embeddings from the penultimate layer, before log_softmax
        class GraphSAGE_Embedder(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.conv1 = original_model.conv1
                self.conv2 = original_model.conv2

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                # Return the output of the second layer before the final classification activation
                x = self.conv2(x, edge_index)
                return x

        embedder_model = GraphSAGE_Embedder(model)
        embeddings = embedder_model(data).detach().cpu().numpy()

    labels = data.y.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random')
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=100, alpha=0.8)
    plt.title('t-SNE visualization of GraphSAGE Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ticks=range(max(labels) + 1), label='Node Class')
    plt.grid(True)
    plt.show()
