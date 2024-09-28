import streamlit as st
import json
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from password_statistics import calculate_entropy
from datasketch import MinHash
import math
import os
import numpy as np
from sklearn.manifold import MDS

# Function to load n-gram probabilities from JSON
def load_ngram_probabilities(ngram_prob_path):
    with open(ngram_prob_path, 'r', encoding='utf-8') as file:
        ngram_probs = json.load(file)
    return ngram_probs

def calculate_ngram_log_likelihood(s, ngram_probs, n=2, smoothing=1e-10):
    log_likelihood_value = 0.0
    len_s = len(s)
    for i in range(len_s - n + 1):
        prev_ngram = s[i:i+n-1]
        next_char = s[i+n-1]
        # Access the nested dictionary directly
        next_char_probs = ngram_probs.get(prev_ngram)
        if next_char_probs is not None:
            prob = next_char_probs.get(next_char, smoothing)
        else:
            prob = smoothing
        log_likelihood_value += math.log(prob)
    return log_likelihood_value

def find_nearest_ngram_cluster(password_ll, ngram_clusters):
    """
    Find the nearest n-gram cluster based on log-likelihood.
    """
    nearest_cluster_key = None
    min_diff = float('inf')
    
    for cluster_key, cluster_info in ngram_clusters.items():
        try:
            cluster_ll = float(cluster_info.get('Average Log Likelihood', 0))
        except (ValueError, TypeError):
            continue
        diff = abs(cluster_ll - password_ll)
        if diff < min_diff:
            min_diff = diff
            nearest_cluster_key = cluster_key
    
    if nearest_cluster_key is None:
        st.error("No n-gram clusters available.")
        return None
    
    return nearest_cluster_key

def find_and_show_closest_ngram_cluster(input_string, clusters, ngram_probs, top_n=10, n=2, smoothing=1e-10):
    input_ll = calculate_ngram_log_likelihood(input_string.lower(), ngram_probs, n, smoothing)
    closest_cluster_key = find_nearest_ngram_cluster(input_ll, clusters)

    if closest_cluster_key is None:
        st.error("No clusters found for the given password.")
        return [], None

    closest_cluster = clusters[closest_cluster_key]

    # Extract passwords from the cluster
    passwords_list = closest_cluster.get('Passwords', [])
    extracted_passwords = []
    for entry in passwords_list:
        if isinstance(entry, dict) and 'Password' in entry:
            extracted_passwords.append(entry['Password'])
        elif isinstance(entry, str):
            extracted_passwords.append(entry)
        else:
            continue

    if not extracted_passwords:
        st.warning("No passwords found in the nearest n-gram cluster.")
        return [], closest_cluster.get('Average Log Likelihood', 0)

    # Find top N closest passwords based on log-likelihood difference
    distances = []
    for password in extracted_passwords:
        password_ll = calculate_ngram_log_likelihood(password.lower(), ngram_probs, n, smoothing)
        ll_diff = abs(input_ll - password_ll)
        distances.append((password, ll_diff))

    # Sort by smallest difference
    distances.sort(key=lambda x: x[1])
    return [password for password, _ in distances[:top_n]], closest_cluster.get('Average Log Likelihood', 0)

def create_ngram_cluster_dataframe(clusters, user_cluster_key, ngram_probs, n=2):
    cluster_data = []
    for cluster_key, cluster_info in clusters.items():
        try:
            cluster_ll = float(cluster_info.get('Average Log Likelihood', 0))
        except (ValueError, TypeError):
            continue
        
        is_user_cluster = (cluster_key == user_cluster_key)

        # Extract sample passwords
        passwords_list = cluster_info.get('Passwords', [])[:3]
        sample_passwords = []
        for entry in passwords_list:
            if isinstance(entry, dict) and 'Password' in entry:
                sample_passwords.append(entry['Password'])
            elif isinstance(entry, str):
                sample_passwords.append(entry)
            else:
                continue

        sample_passwords = sample_passwords if sample_passwords else ['No Passwords']

        cluster_data.append({
            'Log Likelihood': cluster_ll,
            'Cluster Size': len(cluster_info.get('Passwords', [])),
            'User Cluster': is_user_cluster,
            'Sample Passwords': '<br>'.join(sample_passwords)
        })

    df = pd.DataFrame(cluster_data)
    if df.empty:
        st.warning("No clusters available for visualization.")
        return df

    df = df.sort_values(by='Log Likelihood')
    return df

def visualize_ngram_clusters(df, threshold):
    """
    Create a scatter plot for n-gram clusters, highlighting the user's cluster,
    and include the threshold as a vertical line.
    """
    fig = px.scatter(
        df,
        x='Log Likelihood',
        y='Cluster Size',
        title=f"Password Clusters by N-gram Log-Likelihood (Threshold: {threshold:.2f})",
        labels={'Log Likelihood': 'N-gram Log-Likelihood', 'Cluster Size': 'Number of Passwords'},
        size='Cluster Size',
        size_max=20,
        color='User Cluster',
        color_discrete_map={True: 'red', False: 'blue'},
        hover_name='Log Likelihood',
        hover_data={'Cluster Size': True, 'Sample Passwords': True}
    )

    # Update hover template
    fig.update_traces(
        hovertemplate=(
            '<b>N-gram Log-Likelihood:</b> %{x:.2f}<br>'
            '<b>Cluster Size:</b> %{y}<br>'
            '<b>Sample Passwords:</b><br>%{customdata[0]}<br>'
            '<extra></extra>'
        ),
        customdata=df[['Sample Passwords']].values
    )

    # Add a vertical line for the threshold
    fig.add_vline(
        x=threshold,
        line_color='red',
        line_dash="dash",
        annotation_text="Threshold",
        annotation_position="top right"
    )

    return fig

# Function to find the nearest entropy cluster
def find_nearest_entropy_cluster(password_entropy, clusters):
    """Find the nearest cluster by entropy value."""
    nearest_cluster = None
    min_diff = float('inf')

    for entropy in clusters.keys():
        entropy_value = float(entropy)
        diff = abs(entropy_value - password_entropy)
        if diff < min_diff:
            min_diff = diff
            nearest_cluster = entropy_value
    return nearest_cluster

# Show Closest Cluster and Find Top N Closest Passwords
def find_and_show_closest_cluster(input_string, clusters, top_n=10):
    input_entropy = calculate_entropy(input_string)
    closest_cluster_entropy = None
    closest_cluster = None
    min_entropy_diff = float('inf')

    # Step 1: Find the closest cluster by calculating the absolute entropy difference
    for cluster_entropy, cluster_members in clusters.items():
        entropy_diff = abs(input_entropy - float(cluster_entropy))
        if entropy_diff < min_entropy_diff:
            min_entropy_diff = entropy_diff
            closest_cluster_entropy = float(cluster_entropy)
            closest_cluster = cluster_members

    # Step 2: Return the top N closest passwords within the closest cluster
    if closest_cluster is not None:
        distances = []
        for password in closest_cluster:
            password_entropy = calculate_entropy(password)
            entropy_diff = abs(input_entropy - password_entropy)
            distances.append((password, entropy_diff))

        # Sort passwords by entropy difference and return the top N closest ones
        distances.sort(key=lambda x: x[1])
        return [password for password, _ in distances[:top_n]], closest_cluster_entropy

    return [], None  # If no cluster found

# Function to create a MinHash for a given string
def get_minhash(string, num_perm=128):
    """Generate a MinHash signature for the input string."""
    minhash = MinHash(num_perm=num_perm)
    for char in set(string):  # Treat the string as a set of unique characters
        minhash.update(char.encode('utf8'))
    return minhash

# Function to find the closest cluster by MinHash similarity
def find_and_show_closest_minhash_cluster(input_string, clusters, top_n=10, num_perm=128):
    """
    Find the closest cluster by MinHash similarity and return the top N closest passwords.
    """
    input_minhash = get_minhash(input_string, num_perm=num_perm)
    closest_cluster = None
    max_similarity = -1  # Similarity starts from 0 and goes up to 1

    # Step 1: Compare input MinHash with each cluster's MinHash to find the closest one
    for cluster_label, cluster_members in clusters.items():
        if not cluster_members:
            continue  # Skip empty clusters
        # Use the first password to represent the cluster
        cluster_minhash = get_minhash(cluster_members[0], num_perm=num_perm)
        similarity = input_minhash.jaccard(cluster_minhash)
        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_cluster = cluster_members

    # Step 2: Sort passwords within the closest cluster based on MinHash similarity
    distances = []
    if closest_cluster:
        for password in closest_cluster:
            password_minhash = get_minhash(password, num_perm=num_perm)
            similarity = input_minhash.jaccard(password_minhash)
            distances.append((password, similarity))
    
        # Sort passwords by similarity (highest to lowest) and return the top N closest ones
        distances.sort(key=lambda x: x[1], reverse=True)
        return [password for password, _ in distances[:top_n]], max_similarity

    return [], None

def find_closest_minhash_cluster_label(input_string, clusters, num_perm=128):
    """
    Find the label of the closest cluster by MinHash similarity.
    """
    input_minhash = get_minhash(input_string, num_perm=num_perm)
    closest_cluster_label = None
    max_similarity = -1  # Similarity starts from 0 and goes up to 1

    # Step 1: Compare input MinHash with each cluster's MinHash to find the closest one
    for cluster_label, cluster_members in clusters.items():
        if not cluster_members:
            continue  # Skip empty clusters
        # Use the first password to represent the cluster
        cluster_minhash = get_minhash(cluster_members[0], num_perm=num_perm)
        similarity = input_minhash.jaccard(cluster_minhash)
        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_cluster_label = cluster_label

    return closest_cluster_label, max_similarity

def plot_entropy(password, entropy_clusters):
    # Calculate the entropy of the input password
    password_entropy = calculate_entropy(password)

    # Find the nearest entropy cluster and the closest passwords
    nearest_entropy_cluster = find_nearest_entropy_cluster(password_entropy, entropy_clusters)
    closest_passwords, closest_entropy = find_and_show_closest_cluster(password, entropy_clusters)

    # Prepare data for Entropy visualization
    entropy_cluster_data = []
    for entropy, passwords in entropy_clusters.items():
        is_user_cluster = (float(entropy) == nearest_entropy_cluster)
        # Get up to 3 sample passwords
        sample_passwords = passwords[:3] if len(passwords) > 0 else ['No Passwords']
        entropy_cluster_data.append({
            'Entropy': float(entropy),
            'Cluster Size': len(passwords),
            'User Cluster': is_user_cluster,  # Mark the user's cluster
            'Sample Passwords': '<br>'.join(sample_passwords)  # Combine sample passwords with line breaks for hover
        })

    # Create a DataFrame for Entropy Clusters
    df_entropy = pd.DataFrame(entropy_cluster_data)
    df_entropy = df_entropy.sort_values(by='Entropy')

    # Create an interactive scatter plot for Entropy Clusters
    fig_entropy = px.scatter(
        df_entropy,
        x='Entropy',
        y='Cluster Size',
        title="Password Clusters by Entropy",
        labels={'Entropy': 'Entropy Value', 'Cluster Size': 'Number of Passwords'},
        size='Cluster Size',
        size_max=20,
        color='User Cluster',  # Color user's cluster differently
        color_discrete_map={True: 'red', False: 'blue'}  # User's cluster is red, others are blue
    )

    # Update the hover data to include entropy value, cluster size, and sample passwords
    fig_entropy.update_traces(
        hovertemplate=(
            '<b>Entropy:</b> %{x:.2f}<br>'  # Display entropy value
            '<b>Cluster Size:</b> %{y}<br>'  # Display number of passwords
            '<b>Sample Passwords:</b><br>%{customdata[0]}<br>'  # Display sample passwords
            '<extra></extra>'
        ),
        customdata=df_entropy[['Sample Passwords']].values
    )

    # Show the Entropy Cluster plot
    st.plotly_chart(fig_entropy, use_container_width=True)

    # Display the entropy of the input password and the assigned cluster
    st.write(f"Your password's entropy: {password_entropy:.2f}")
    st.write(f"Assigned to entropy cluster: {nearest_entropy_cluster:.2f}")

    # Display the top 10 closest passwords by entropy
    st.subheader("Top 10 Closest Passwords by Entropy")
    if closest_passwords:
        for i, p in enumerate(closest_passwords, 1):
            st.write(f"{i}. {p}")
    else:
        st.write("No similar passwords found in the nearest entropy cluster.")

def plot_minhash(password, minhash_clusters):
    # Find the closest MinHash cluster and get the 10 closest passwords
    closest_passwords_minhash, max_similarity_minhash = find_and_show_closest_minhash_cluster(
        password, minhash_clusters
    )

    # Display the MinHash similarity and the closest passwords
    st.write(f"Max MinHash similarity: {max_similarity_minhash:.2f}")
    
    st.subheader("Top 10 Closest Passwords by MinHash Similarity")
    if closest_passwords_minhash:
        for i, p in enumerate(closest_passwords_minhash, 1):
            st.write(f"{i}. {p}")
    else:
        st.write("No similar passwords found in the nearest MinHash cluster.")

def plot_ngram(password, ngram_clusters_json, ngram_probs):
    # Get threshold from the JSON data
    threshold = ngram_clusters_json.get('Threshold', None)
    ngram_clusters = ngram_clusters_json.get('Clusters', {})

    # Define n for n-grams
    n = 2  # Adjust if necessary

    # Calculate n-gram log-likelihood and find the nearest n-gram cluster
    password_ngram_ll = calculate_ngram_log_likelihood(password, ngram_probs, n, smoothing=1e-10)
    nearest_ngram_cluster_key = find_nearest_ngram_cluster(password_ngram_ll, ngram_clusters)
    closest_ngram_passwords, closest_ngram_ll = find_and_show_closest_ngram_cluster(
        password, ngram_clusters, ngram_probs, top_n=10, n=2, smoothing=1e-10
    )

    # Prepare data for n-gram visualization
    ngram_df = create_ngram_cluster_dataframe(ngram_clusters, nearest_ngram_cluster_key, ngram_probs, n=2)

    # Check if the DataFrame is not empty before plotting
    if not ngram_df.empty:
        # Visualize n-gram clusters, including the threshold
        ngram_fig = visualize_ngram_clusters(ngram_df, threshold)
        st.plotly_chart(ngram_fig, use_container_width=True)
    else:
        st.warning("No n-gram clusters available for visualization.")

    # Display the n-gram log-likelihood of the input password and the assigned cluster
    st.write(f"Your password's n-gram log-likelihood: {password_ngram_ll:.2f}")
    st.write(f"Assigned to n-gram cluster with Average Log Likelihood: {float(closest_ngram_ll):.2f}")

    # Display the top 10 closest passwords by n-gram log-likelihood
    st.subheader("Top 10 Closest Passwords by N-gram Log-Likelihood")
    if closest_ngram_passwords:
        for i, p in enumerate(closest_ngram_passwords, 1):
            st.write(f"{i}. {p}")
    else:
        st.write("No similar passwords found in the nearest n-gram cluster.")

# New functions for loading and visualizing clusters with similarities
def load_clusters_and_similarities(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    clusters = data['Clusters']
    similarities = data['Cluster Similarities']
    return clusters, similarities

def get_top_k_clusters(clusters, similarities, top_k=10, include_clusters=None):
    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    # Ensure include_clusters are in the list
    if include_clusters is not None:
        # Move included clusters to the front
        included = [item for item in sorted_clusters if item[0] in include_clusters]
        others = [item for item in sorted_clusters if item[0] not in include_clusters]
        sorted_clusters = included + others

    # Take the top K clusters
    top_clusters = sorted_clusters[:top_k]

    # Now, extract the data
    cluster_names = [cluster[0] for cluster in top_clusters]
    cluster_sizes = np.array([len(cluster[1]) for cluster in top_clusters])
    cluster_examples = [cluster[1][:3] for cluster in top_clusters]

    # Build the similarity matrix
    num_clusters = len(cluster_names)
    similarity_matrix = np.ones((num_clusters, num_clusters))

    for i, cluster_i in enumerate(cluster_names):
        for j, cluster_j in enumerate(cluster_names):
            if i != j:
                pair_key = f"{cluster_i} vs {cluster_j}"
                similarity = similarities.get(pair_key, None)
                if similarity is not None:
                    similarity_matrix[i, j] = 1 - similarity  # Convert similarity to distance
                else:
                    similarity_matrix[i, j] = 1  # Max distance if similarity not found
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

def visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix, user_cluster_label=None):
    # Perform multidimensional scaling (MDS) to convert similarity matrix to 2D positions
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(similarity_matrix)

    # Calculate marker sizes so that marker areas are proportional to cluster sizes
    scaling_factor = 2  # Adjust this value to scale marker sizes for better visualization
    marker_sizes = np.sqrt(cluster_sizes) * scaling_factor

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'Cluster Name': cluster_names,
        'Cluster Size': cluster_sizes,
        'Examples': cluster_examples,
    })

    # Determine colors
    plot_df['Color'] = plot_df['Cluster Name'].apply(
        lambda x: 'User Cluster' if x == user_cluster_label else 'Other Clusters'
    )

    # Create hover text
    plot_df['Hover Text'] = plot_df.apply(
        lambda row: f"<b>{row['Cluster Name']}</b><br>Size: {row['Cluster Size']}<br><b>Examples:</b><br>{'<br>'.join(row['Examples'])}",
        axis=1
    )

    # Create a scatter plot using Plotly Express
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        size='Cluster Size',
        size_max=20,
        color='Color',
        color_discrete_map={'User Cluster': 'red', 'Other Clusters': 'blue'},
        hover_name='Cluster Name',
        hover_data={'Cluster Size': True, 'Examples': True},
    )

    # Update hover template to use the custom hover text
    fig.update_traces(
        hovertemplate='%{customdata}<extra></extra>',
        customdata=plot_df['Hover Text']
    )

    # Update layout for aesthetics
    fig.update_layout(
        title="Visualization of Top Clusters by Size and Similarity",
        xaxis_title="MDS Dimension 1",
        yaxis_title="MDS Dimension 2",
        template="plotly_white",
        width=900,
        height=700,
        showlegend=False  # Hide legend if desired
    )

    # Show the interactive plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)

def dynamic_clusters_page():
    # Page for inserting a password
    st.header('Insert Password for Clustering Analysis')
    st.write("Please enter your password below.")
    password = st.text_input("Enter a password for analysis",
                             help="Type your password here to see how it compares to common length patterns")
    st.caption("Your password will be analyzed locally and not stored or transmitted.")

    if not password:
        st.write("Enter a password to analyze.")
        return

    json_files_path = ''
    entropy_json_name = 'entropy_clusters.json'
    minhash_json_name = 'minhash_clusters.json'
    ngram_prob_name = 'ngram_probs.json'
    ngram_clusters_name = 'ngram_clusters.json'
    minhash_similarity_json_name = 'minhash_clusters_with_similarity.json' 

    entropy_json_path = os.path.join(json_files_path, entropy_json_name)
    ngram_clusters_path = os.path.join(json_files_path, ngram_clusters_name)
    ngram_prob_path = os.path.join(json_files_path, ngram_prob_name)
    minhash_json_path = os.path.join(json_files_path, minhash_json_name)
    minhash_similarity_json_path = os.path.join(json_files_path, minhash_similarity_json_name)

    # Load JSON data
    with open(entropy_json_path, 'r', encoding='utf-8') as json_file:
        entropy_clusters = json.load(json_file)

    with open(minhash_json_path, 'r', encoding='utf-8') as json_file:
        minhash_clusters = json.load(json_file)

    with open(ngram_clusters_path, 'r', encoding='utf-8') as json_file:
        ngram_clusters_json = json.load(json_file)

    """ Entropy Clustering """
    plot_entropy(password, entropy_clusters)

    """ N-gram Clustering """
    # Load n-gram probabilities
    ngram_probs = load_ngram_probabilities(ngram_prob_path)
    plot_ngram(password, ngram_clusters_json, ngram_probs)

    """ MinHash Clustering """
    st.subheader("MinHash Clusters Visualization")

    # Let the user select the number of top clusters to display
    top_k = st.slider('Select number of top clusters to display', min_value=5, max_value=50, value=10, step=5)

    # Load clusters and similarities
    if os.path.exists(minhash_similarity_json_path):
        clusters, similarities = load_clusters_and_similarities(minhash_similarity_json_path)

        # Find the closest MinHash cluster to the user's password
        closest_cluster_label, max_similarity = find_closest_minhash_cluster_label(password, clusters)

        if max_similarity == 0:
            # No similar cluster found, create a new cluster for the user's password
            user_cluster_label = 'User Password'
            clusters[user_cluster_label] = [password]

            # Get top clusters without including any specific clusters
            cluster_names, cluster_sizes, cluster_examples, similarity_matrix = get_top_k_clusters(
                clusters, similarities, top_k=top_k)

            # Add the user's cluster to the data
            cluster_names.append(user_cluster_label)
            cluster_sizes = np.append(cluster_sizes, 1)
            cluster_examples.append([password])

            # Compute similarities between user's password and existing clusters
            user_similarities = []
            input_minhash = get_minhash(password)
            for cluster_label in cluster_names[:-1]:  # Exclude the user's cluster itself
                cluster_minhash = get_minhash(clusters[cluster_label][0])
                sim = input_minhash.jaccard(cluster_minhash)
                user_similarities.append(1 - sim)  # Distance

            # Now, update the similarity_matrix to include the new cluster
            new_row = np.array(user_similarities + [0])  # Similarity with self is 0
            similarity_matrix = np.vstack([similarity_matrix, new_row[:-1]])
            new_col = np.append(new_row, 0)[:, np.newaxis]
            similarity_matrix = np.hstack([similarity_matrix, new_col])

            # Update the closest cluster label to 'User Password' to color it differently
            closest_cluster_label = user_cluster_label
        else:
            # Ensure the closest cluster is included
            cluster_names, cluster_sizes, cluster_examples, similarity_matrix = get_top_k_clusters(
                clusters, similarities, top_k=top_k, include_clusters=[closest_cluster_label])

        # Now, proceed to visualize
        visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix, closest_cluster_label)
    else:
        st.warning("MinHash clusters with similarities file not found.")

    plot_minhash(password, minhash_clusters)

# Call the main function to display the page
if __name__ == "__main__":
    dynamic_clusters_page()
