import json
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import MDS

# Function to load clusters and similarities from the JSON file
def load_clusters_and_similarities(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    clusters = data['Clusters']
    similarities = data['Cluster Similarities']
    return clusters, similarities

# Function to extract the top K largest clusters and their similarities
def get_top_k_clusters(clusters, similarities, top_k=10):
    # Sort clusters by size and take the top K largest
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:top_k]
    
    # Get the cluster names, sizes, and example passwords
    cluster_names = [cluster[0] for cluster in sorted_clusters]
    cluster_sizes = np.array([len(cluster[1]) for cluster in sorted_clusters])
    cluster_examples = [cluster[1][:3] for cluster in sorted_clusters]  # Get first 3 passwords as examples

    # Extract the relevant similarities between the top K clusters
    similarity_matrix = np.ones((top_k, top_k))  # Initialize similarity matrix with ones (maximum distance)
    
    for i, cluster_i in enumerate(cluster_names):
        for j, cluster_j in enumerate(cluster_names):
            if i != j:
                pair_key = f"{cluster_i} vs {cluster_j}"
                similarity = similarities.get(pair_key, None)
                if similarity is not None:
                    similarity_matrix[i, j] = 1 - similarity  # Convert similarity to distance
                else:
                    similarity_matrix[i, j] = 1  # Assume maximum distance if similarity not found
    
    # Make the similarity matrix symmetric and ensure diagonal is 0
    similarity_matrix = make_symmetric(similarity_matrix)

    return cluster_names, cluster_sizes, cluster_examples, similarity_matrix

# Function to ensure the matrix is symmetric and properly formatted for MDS
def make_symmetric(matrix):
    """Ensure the matrix is symmetric."""
    sym_matrix = np.copy(matrix)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            sym_matrix[j, i] = sym_matrix[i, j]
    # Ensure diagonal is 0 (as it represents the distance of a cluster with itself)
    np.fill_diagonal(sym_matrix, 0)
    return sym_matrix

# Function to visualize the clusters using MDS and Plotly
def visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix):
    # Perform multidimensional scaling (MDS) to convert similarity matrix to 2D positions
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(similarity_matrix)

    # Calculate marker sizes so that marker areas are proportional to cluster sizes
    # Since area ∝ (size / 2)^2, we set size ∝ sqrt(cluster_size)
    scaling_factor = 2  # Adjust this value to scale marker sizes for better visualization
    marker_sizes = np.sqrt(cluster_sizes) * scaling_factor

    # Create hover text including cluster name, size, and example passwords
    hover_text = []
    for name, size, examples in zip(cluster_names, cluster_sizes, cluster_examples):
        example_passwords = "<br>".join(examples)
        text = f"<b>{name}</b><br>Size: {size}<br><b>Examples:</b><br>{example_passwords}"
        hover_text.append(text)

    # Create a scatter plot using Plotly
    fig = go.Figure()

    # Add scatter points for clusters
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers',
        marker=dict(
            size=marker_sizes,  # Marker sizes proportional to sqrt(cluster_sizes)
            color=cluster_sizes,        # Color based on cluster sizes
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cluster Size")
        ),
        text=hover_text,  # Add hover text
        hoverinfo='text',  # Display text on hover
    ))

    # Update layout for aesthetics
    fig.update_layout(
        title="Visualization of Top 10 Clusters by Size and Similarity",
        xaxis_title="MDS Dimension 1",
        yaxis_title="MDS Dimension 2",
        template="plotly_white",
        width=900,
        height=700
    )

    # Show the interactive plot
    fig.show()

# Main function to load data and visualize the clusters
if __name__ == "__main__":
    # Load the clusters and similarities from the JSON file
    json_file = 'minhash_clusters_with_similarity.json'
    clusters, similarities = load_clusters_and_similarities(json_file)

    # Get the top 10 clusters and their similarity matrix
    cluster_names, cluster_sizes, cluster_examples, similarity_matrix = get_top_k_clusters(
        clusters, similarities, top_k=200)

    # Visualize the clusters
    visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix)
