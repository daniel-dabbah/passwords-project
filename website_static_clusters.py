import streamlit as st
import json
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.manifold import MDS  # Added for multidimensional scaling

def plot_entropy_vs_likelihood_by_cluster(cluster_json):
    # Load the cluster data from the JSON file
    with open(cluster_json, 'r', encoding='utf-8') as json_file:
        cluster_data = json.load(json_file)

    # Extract average entropy and log-likelihood values for each cluster
    entropy_likelihood_data = []
    for cluster_key, cluster_info in cluster_data['Clusters'].items():
        avg_likelihood = cluster_info['Average Log Likelihood']
        avg_entropy = cluster_info['Average Entropy']
        classification = cluster_info['Classification']
        cluster_size = len(cluster_info['Passwords'])  # Use cluster size for size scaling
        
        # Filter: Only include clusters with size > 100, entropy < 200, and log-likelihood > -500
        if cluster_size > 100 and avg_entropy < 200 and avg_likelihood > -500:
            entropy_likelihood_data.append({
                'Cluster': cluster_key,
                'Average Log Likelihood': avg_likelihood,
                'Average Entropy': avg_entropy,
                'Classification': classification,
                'Cluster Size': cluster_size
            })

    # Create a DataFrame for plotting
    df = pd.DataFrame(entropy_likelihood_data)

    # Create a scatter plot for average entropy vs average log-likelihood at the cluster level
    fig = px.scatter(df, x='Average Log Likelihood', y='Average Entropy', color='Classification',
                     title='Relationship between Average Log-Likelihood and Entropy',
                     labels={'Average Log Likelihood': 'Average Log Likelihood', 'Average Entropy': 'Average Entropy'},
                     size='Cluster Size',  # Use cluster size to scale marker size
                     color_discrete_map={'Meaningful': 'green', 'Gibberish': 'red'})

    # Display the plot in Streamlit
    st.plotly_chart(fig)


def plot_entropy_clusters(entropy_clusters):
    # Prepare data for visualization
    cluster_data = []
    for entropy, passwords in entropy_clusters.items():
        # Filter: Only include clusters with entropy < 200 and size > 100
        if float(entropy) < 200 and len(passwords) > 100:
            sample_passwords = passwords[:3]  # Get up to 3 sample passwords
            cluster_data.append({
                'Entropy': float(entropy),
                'Cluster Size': len(passwords),
                'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
                'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
                'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
            })

    # Create a DataFrame for Plotly
    df = pd.DataFrame(cluster_data)

    # Sort DataFrame by Entropy in ascending order
    df = df.sort_values(by='Entropy')

    # Create an interactive scatter plot colored by cluster size
    fig = px.scatter(df, x='Entropy', y='Cluster Size',
                     title="Password Clusters by Entropy (Filtered)",
                     labels={'Entropy': 'Entropy Value', 'Cluster Size': 'Number of Passwords'},
                     size='Cluster Size', size_max=20,
                     color='Cluster Size',  # Color by cluster size
                     color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_traces(
        hovertemplate=(
            '<b>Entropy:</b> %{x:.2f}<br>'
            '<b>Cluster Size:</b> %{y}<br>'
            '<b>Sample Passwords:</b><br>%{hovertext}'
            '<extra></extra>'
        ),
        hovertext=df[['Sample Password 1', 'Sample Password 2', 'Sample Password 3']].apply(lambda x: '<br>'.join(x), axis=1)
    )

    st.plotly_chart(fig)

    st.write("""
             The scatter plot above visualizes the clustering of passwords based on their entropy values. \n
                Each point represents a cluster of passwords with a specific entropy value. 
                The size of the point corresponds to the number of passwords in the cluster. 
                Hover over the points to view the entropy value, cluster size, and sample passwords.  \n

                The table below provides a detailed view of the clusters, including the entropy value, cluster size, and sample passwords. 
                The sample passwords are representative of the passwords in each cluster. 
                The clusters in the table are sorted by entropy in ascending order. 
                You can explore the distribution of password entropy values and the corresponding cluster sizes. \n
             
                It can be seen that most passwords fall within the lower entropy range, indicating weaker password strength. \n
        """)

    st.subheader('Entropy Clusters Table')
    st.dataframe(df[['Entropy', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])


def plot_ngram_clusters(ngram_clusters):
    # Extract threshold
    threshold = ngram_clusters['Threshold']

    # Prepare data for n-gram visualization
    ngram_data_list = []
    for log_likelihood, cluster in ngram_clusters['Clusters'].items():
        average_likelihood = cluster['Average Log Likelihood']
        classification = cluster['Classification']
        passwords = cluster['Passwords']

        # Filter: Only include clusters with log-likelihood > -500 and size > 100
        if average_likelihood > -500 and len(passwords) > 100:
            # Extract up to 3 sample passwords
            sample_passwords = [p['Password'] for p in passwords[:3]]
            ngram_data_list.append({
                'Log Likelihood': float(log_likelihood),
                'Average Log Likelihood': average_likelihood,
                'Classification': classification,
                'Cluster Size': len(passwords),
                'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
                'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
                'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
            })

    # Create a DataFrame for Plotly
    ngram_df = pd.DataFrame(ngram_data_list)
    ngram_df = ngram_df.sort_values(by='Log Likelihood', ascending=False)

    # Create an interactive scatter plot for n-gram clusters
    ngram_fig = px.scatter(
        ngram_df, 
        x='Log Likelihood', 
        y='Cluster Size',
        title=f"Password Clusters by N-gram Log Likelihood (Threshold: {threshold:.2f})",
        labels={'Log Likelihood': 'Log Likelihood Value', 'Cluster Size': 'Number of Passwords'},
        size='Cluster Size', 
        size_max=20,
        color='Cluster Size',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Update the hover template to show only average log-likelihood and 3 sample passwords
    ngram_fig.update_traces(
        hovertemplate=(
            '<b>Average Log Likelihood:</b> %{customdata[0]:.2f}<br>'
            '<b>Cluster Size:</b> %{y}<br>'
            '<b>Sample Passwords:</b><br>'
            '- %{customdata[1]}<br>'
            '- %{customdata[2]}<br>'
            '- %{customdata[3]}<br>'
            '<extra></extra>'
        ),
        customdata=ngram_df[['Average Log Likelihood', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']].values
    )

    # Add a vertical line for the threshold
    ngram_fig.add_vline(
        x=threshold, 
        line_color='red', 
        line_dash="dash", 
        annotation_text="Threshold", 
        annotation_position="top right"
    )

    # Display the Plotly chart
    st.plotly_chart(ngram_fig)

    # Display the clusters in a table with separate columns for the sample passwords
    st.subheader('N-gram Clusters Table')
    st.dataframe(ngram_df[['Average Log Likelihood', 'Classification', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])


def plot_minhash_clusters(minhash_clusters):
    # Prepare data for MinHash clusters visualization
    minhash_cluster_data = []
    for cluster_label, passwords in minhash_clusters.items():
        cluster_size = len(passwords)
        # Filter clusters: only include clusters with a size smaller than 100K
        if cluster_size < 100000 and cluster_size >= 10:
            # Get up to 3 sample passwords for the table
            sample_passwords = passwords[:3]
            minhash_cluster_data.append({
                'Cluster Label': cluster_label,
                'Cluster Size': cluster_size,
                'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
                'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
                'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
            })

    # Check if there are any clusters to display
    if len(minhash_cluster_data) == 0:
        st.write("No clusters found.")
        return

    # Create a DataFrame for MinHash clusters
    minhash_df = pd.DataFrame(minhash_cluster_data)

    # Display only the clusters and sample passwords in a table
    st.subheader('MinHash Clusters Table')
    st.dataframe(minhash_df[['Cluster Label', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])

# New functions for loading and visualizing clusters with similarities
def load_clusters_and_similarities(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    clusters = data['Clusters']
    similarities = data['Cluster Similarities']
    return clusters, similarities

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

def make_symmetric(matrix):
    """Ensure the matrix is symmetric."""
    sym_matrix = np.copy(matrix)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            sym_matrix[j, i] = sym_matrix[i, j]
    # Ensure diagonal is 0 (as it represents the distance of a cluster with itself)
    np.fill_diagonal(sym_matrix, 0)
    return sym_matrix

def visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix):
    # Filter out clusters with size 100K or larger
    filtered_cluster_names = []
    filtered_cluster_sizes = []
    filtered_cluster_examples = []
    for name, size, examples in zip(cluster_names, cluster_sizes, cluster_examples):
        if size < 100_000:  # Only include clusters smaller than 100K
            filtered_cluster_names.append(name)
            filtered_cluster_sizes.append(size)
            filtered_cluster_examples.append(examples)

    # Check if any clusters remain after filtering
    if len(filtered_cluster_names) == 0:
        st.warning("No clusters found with size smaller than 100K.")
        return

    # Perform multidimensional scaling (MDS) to convert similarity matrix to 2D positions
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(similarity_matrix[:len(filtered_cluster_names), :len(filtered_cluster_names)])

    # Calculate marker sizes so that marker areas are proportional to cluster sizes
    scaling_factor = 0.5  # Adjust this value to scale marker sizes for better visualization
    marker_sizes = np.sqrt(filtered_cluster_sizes) * scaling_factor

    # Create hover text including cluster name, size, and example passwords
    hover_text = []
    for name, size, examples in zip(filtered_cluster_names, filtered_cluster_sizes, filtered_cluster_examples):
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
            color=filtered_cluster_sizes,  # Color based on filtered cluster sizes
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cluster Size")
        ),
        text=hover_text,  # Add hover text
        hoverinfo='text',  # Display text on hover
    ))

    # Update layout for aesthetics
    fig.update_layout(
        title="Visualization of Top Clusters by Size and Similarity",
        xaxis_title="MDS Dimension 1",
        yaxis_title="MDS Dimension 2",
        template="plotly_white",
        width=900,
        height=700
    )

    # Show the interactive plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)



def static_clusters_page():

    json_files_path = 'json_files'
    entropy_json_name = 'entropy_clusters.json'
    ngram_json_name = 'ngram_clusters.json'
    minhash_json_name = 'minhash_clusters.json'
    minhash_similarity_json_name = 'minhash_clusters_with_similarity.json'

    entropy_json_path = os.path.join(json_files_path, entropy_json_name)
    ngram_json_path = os.path.join(json_files_path, ngram_json_name)
    minhash_json_path = os.path.join(json_files_path, minhash_json_name)
    minhash_similarity_json_path = os.path.join(json_files_path, minhash_similarity_json_name)

    # Load the entropy dictionary from the JSON file
    with open(entropy_json_path, 'r', encoding='utf-8') as json_file:
        entropy_clusters = json.load(json_file)

    # Load the n-gram clustering data from the JSON file
    with open(ngram_json_path, 'r', encoding='utf-8') as json_file:
        ngram_clusters = json.load(json_file)

    # Load the MinHash clusters from the JSON file
    with open(minhash_json_path, 'r', encoding='utf-8') as json_file:
        minhash_clusters = json.load(json_file)

    st.header('Clustering Analysis')
    st.write("""
        This page provides an overview of password clustering methods based on three key metrics: **Entropy**, **N-gram Log-Likelihood**, and **MinHash Similarity**.
        These clustering methods allow us to group passwords that exhibit similar characteristics, offering insights into password strength, predictability, and similarity.
        Passwords are grouped into clusters based on these metrics, and the visualizations here show how passwords are distributed across these clusters.
    """)

    """ Entropy Clustering """
    st.header('Clustering by Entropy')
    st.write("""
        **Entropy** is a measure of unpredictability or randomness in a password. A password with higher entropy is harder to guess because it incorporates a larger variety of characters or a longer sequence of characters.
        - **Formula for Entropy**: Entropy = Password Length × log₂(Character Set Size)
        - **Character Set Size**: This refers to the number of possible characters used in a password, including lowercase letters, uppercase letters, numbers, and symbols.
        
        For example, an 8-character password with lowercase, uppercase, and numbers (a total of 62 possible characters) has an entropy of approximately 47.63 bits.
        Passwords with entropy values above 80 bits are considered secure, typically requiring a mix of all character types and a minimum length of 12 characters.
        
        In this clustering analysis, we group passwords based on their entropy values. Passwords with a small difference in entropy are clustered together, helping visualize where passwords stand in terms of complexity compared to others.
    """)
    plot_entropy_clusters(entropy_clusters)

    """ N-Gram Clustering """
    st.header('Clustering by N-gram Log-Likelihood')
    st.write("""
        **N-gram Log-Likelihood** evaluates the probability of a sequence of characters based on patterns learned from a large dataset of common passwords.
        - **N-grams** refer to contiguous sequences of characters, and in this analysis, we focus on bi-grams (sequences of 2 characters).
        - **Log-Likelihood** is the logarithmic probability of these character sequences occurring based on observed patterns from common passwords.
        
        Passwords with high log-likelihood values are considered **meaningful**, as they follow common patterns and are more predictable. Conversely, passwords with low log-likelihood values are classified as **gibberish**, as they are more random and harder to predict.

        - **Meaningful passwords** are those that exhibit common patterns and linguistic structures, making them vulnerable to attacks that exploit dictionaries or language models.
        - **Gibberish passwords** appear more random and unpredictable, providing stronger protection against such attacks.
        
        In this clustering method, we calculate the n-gram log-likelihood of each password and group them into clusters. A **threshold** is used to distinguish between meaningful and gibberish clusters, and the visualization shows the distribution of passwords based on this metric.
             We can observe how passwords are grouped based on their structural predictability and linguistic patterns.
             Notice that the clusters that are classified as meaningful are also the bigger clusters, indicating that the datset contains a lot of common passwords.
    """)
    plot_ngram_clusters(ngram_clusters)

    """ MinHash Clustering """
    st.header('Clustering by MinHash')
    st.write("""
        **MinHash** is a method used to estimate the similarity between sets of characters in passwords. By hashing the character sets of passwords, we can efficiently compare the similarity of passwords and group them into clusters based on shared characters.
        - Each password is hashed into a compact **MinHash signature**, representing its character set.
        - **Similarity** between passwords is calculated based on the overlap of their character sets.

        Passwords with similar MinHash signatures are grouped together, even if their sequences or lengths differ. This clustering technique helps identify passwords that are structurally similar based on the characters they use, rather than their exact sequence.
        
        The scatter plot visualizes these clusters, and the closer the points, the more similar the passwords are. This method is particularly effective for large datasets, as it provides a fast and scalable way to estimate password similarity.
    """)
    plot_minhash_clusters(minhash_clusters)

    """ MinHash Clusters Visualization with Similarity """
    st.subheader('MinHash Clusters Visualization with Similarity')
    st.write("""
        In addition to clustering passwords based on their MinHash signatures, we also visualize the **similarity** between clusters. Clusters that are positioned closer together share more similarities, and the size and color of each cluster reflect its size.
        
        This visualization allows us to interpret the relationships between password clusters and helps us see how common or unique different groups of passwords are.
    """)
    
    # Let the user select the number of top clusters to display
    top_k = st.slider('Select number of top clusters to display', min_value=5, max_value=100, value=10, step=5)

    if os.path.exists(minhash_similarity_json_path):
        clusters, similarities = load_clusters_and_similarities(minhash_similarity_json_path)

        cluster_names, cluster_sizes, cluster_examples, similarity_matrix = get_top_k_clusters(
            clusters, similarities, top_k=top_k)

        visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix)
    else:
        st.warning("MinHash clusters with similarities file not found.")

    """ Entropy vs Log-Likelihood """
    st.header('Entropy vs Log-Likelihood')
    st.write("""
        This section explores the relationship between the **entropy** and **log-likelihood** of password clusters. We observe that clusters with higher entropy tend to have lower log-likelihood values, reflecting a balance between password randomness and predictability.
        
        The scatter plot below highlights this relationship, showing how clusters with high entropy often have lower structural predictability, making them more secure.
    """)
    plot_entropy_vs_likelihood_by_cluster(ngram_json_path)

# Call the main function to display the page
if __name__ == "__main__":
    static_clusters_page()

