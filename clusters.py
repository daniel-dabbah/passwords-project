import string
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import password_statistics as ps
import json
import Levenshtein as lev  # Import the Levenshtein library
from datasketch import MinHash, MinHashLSH  # Import the MinHash and MinHashLSH classes
import random

""" Cluster strings by entropy """
# Cluster strings based on entropy similarity
def cluster_strings_by_entropy(strings, entropy_threshold=0.5):
    """
    Cluster strings based on entropy similarity and return a dictionary with the clusters.
    Also save the clusters with more than a certain number of members to a file.
    Args:
        strings (list): List of password strings.
        entropy_threshold (float): Threshold for entropy similarity to cluster passwords.
    
    Returns:
        clusters (dict): Dictionary with entropy values as keys and lists of passwords as values.
    """
    clusters = defaultdict(list)

    # Iterate through the list of strings and calculate entropy for clustering
    for i, string in tqdm(enumerate(strings), total=len(strings), desc="Clustering strings by entropy"):
        entropy = ps.calculate_entropy(string)  # Assuming this function exists and works properly

        # Try to place the string in an existing cluster based on entropy threshold
        found_cluster = False
        for cluster_entropy, cluster_members in clusters.items():
            if abs(entropy - cluster_entropy) <= entropy_threshold:
                clusters[cluster_entropy].append(string)
                found_cluster = True
                break

        # If no cluster is found, create a new one
        if not found_cluster:
            clusters[entropy].append(string)

        # Save clusters to a file with optional filtering for cluster size
        def save_clusters_to_file(clusters, min_size=10, output_file='clusters_by_entropy_output.txt'):
            """
            Save clusters with more than `min_size` members to a file.
            Args:
                clusters (dict): Dictionary of clusters with entropy values as keys and lists of passwords as values.
                min_size (int): Minimum number of members required for a cluster to be saved to the file.
                output_file (str): Path to the output file.
            """
            with open(output_file, 'w', encoding='utf-8') as file:
                for entropy_value, cluster_members in clusters.items():
                    if len(cluster_members) >= min_size:
                        file.write(f"\nCluster (entropy={entropy_value:.2f}, size={len(cluster_members)}):\n")
                        for member in cluster_members:
                            file.write(f"  {member}\n")
            print(f"Clusters with more than {min_size} members saved to {output_file}")
        
        # Function to save clusters to a JSON file
    def save_clusters_to_json(clusters, output_file='entropy_clusters.json'):
        """
        Save the clusters to a JSON file.
        
        Args:
            clusters (dict): Dictionary of clusters with entropy values as keys and lists of passwords as values.
            output_file (str): Path to the output JSON file.
        """
        # Convert the dictionary to have string keys since JSON keys must be strings
        clusters_serializable = {str(key): value for key, value in clusters.items()}

        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(clusters_serializable, json_file, ensure_ascii=False, indent=4)
        print(f"Clusters saved to {output_file}")
    
    save_clusters_to_file(clusters, min_size=10, output_file='clusters_by_entropy_output.txt')
    save_clusters_to_json(clusters, output_file='entropy_clusters.json')

    return clusters

""" Show Closest Cluster """
def find_and_show_closest_cluster(input_string, clusters, top_n=10):
    """
    Find the closest cluster by entropy (without a threshold) and display its members.
    Args:
        input_string (str): The input password to compare.
        clusters (dict): The dictionary of clusters with entropy values as keys and lists of passwords as values.
        top_n (int): The number of closest passwords to return from the closest cluster.
    
    Returns:
        list: A list of the top N closest passwords from the closest cluster by entropy.
    """
    input_entropy = ps.calculate_entropy(input_string)
    closest_cluster_entropy = None
    closest_cluster = None
    min_entropy_diff = float('inf')

    # Step 1: Find the closest cluster by calculating the absolute entropy difference
    for cluster_entropy, cluster_members in clusters.items():
        entropy_diff = abs(input_entropy - cluster_entropy)
        if entropy_diff < min_entropy_diff:
            min_entropy_diff = entropy_diff
            closest_cluster_entropy = cluster_entropy
            closest_cluster = cluster_members

    # Step 2: Display the closest cluster
    if closest_cluster is not None:
        print(f"\nClosest cluster (entropy={closest_cluster_entropy:.2f}, size={len(closest_cluster)}):")
        for password in closest_cluster[:top_n]:
            print(f"  {password}")
    else:
        print(f"No cluster found for the input string: '{input_string}'.")

    # Step 3: Return the top N closest passwords within the closest cluster
    distances = []
    for password in closest_cluster:
        password_entropy = ps.calculate_entropy(password)
        entropy_diff = abs(input_entropy - password_entropy)
        distances.append((password, entropy_diff))

    # Sort passwords by entropy difference and return the top N closest ones
    distances.sort(key=lambda x: x[1])
    return [password for password, _ in distances[:top_n]]


# Function to create a MinHash for a given string
def get_minhash(string, num_perm=128):
    """Create a MinHash for a string."""
    minhash = MinHash(num_perm=num_perm)
    for char in set(string):  # Treat string as a set of unique characters
        minhash.update(char.encode('utf8'))
    return minhash

# Function to calculate gradual similarity between clusters using MinHash with progress bars
def compute_cluster_similarity_top_k(clusters, num_perm=128, top_k=100):
    """Compute the MinHash Jaccard similarity between the top K largest clusters."""
    # Step 1: Sort clusters by size and select the top K
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:top_k]
    
    cluster_similarity = {}
    cluster_keys = [key for key, _ in sorted_clusters]  # Just get the keys of the top K clusters

    # Step 2: Iterate over all top K cluster pairs and compute similarity
    for i in tqdm(range(len(cluster_keys)), desc="Processing outer clusters"):
        cluster_i = cluster_keys[i]
        cluster_i_minhash = get_minhash(clusters[cluster_i][0], num_perm)  # Use first password as cluster representative

        for j in tqdm(range(i + 1, len(cluster_keys)), desc=f"Processing inner clusters for {cluster_i}", leave=False):
            cluster_j = cluster_keys[j]
            cluster_j_minhash = get_minhash(clusters[cluster_j][0], num_perm)

            # Compute MinHash (Jaccard) similarity between the two clusters
            similarity = cluster_i_minhash.jaccard(cluster_j_minhash)

            # Save the similarity score
            cluster_similarity[f"{cluster_i} vs {cluster_j}"] = similarity

    return cluster_similarity

# Function to cluster strings using MinHashLSH and compute gradual similarity between clusters
def cluster_strings_with_minhash_lsh(strings, threshold=0.9, num_perm=128, top_k=100):
    """Cluster strings based on MinHashLSH and compute gradual cluster similarities."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    clusters = defaultdict(list)
    cluster_positions = {}
    similarity_threshold = 0.2  # Define a threshold for closeness in cluster placement

    # Assign random initial positions for clusters (x, y coordinates)
    cluster_positions = {}

    for i, string in tqdm(enumerate(strings), total=len(strings), desc="Clustering strings with MinHashLSH"):
        minhash = get_minhash(string, num_perm)

        # Query for existing clusters in the LSH structure
        existing_cluster = lsh.query(minhash)
        if existing_cluster:
            # If similar, add to the first matching cluster
            cluster_key = existing_cluster[0]
            clusters[cluster_key].append(string)
        else:
            # If no similar cluster, create a new one
            cluster_key = f'cluster_{len(clusters) + 1}'
            clusters[cluster_key].append(string)
            lsh.insert(cluster_key, minhash)

            # Assign random initial positions for new clusters
            cluster_positions[cluster_key] = (random.uniform(-1, 1), random.uniform(-1, 1))

    # Compute the gradual similarity between the top K largest clusters
    cluster_similarity = compute_cluster_similarity_top_k(clusters, num_perm=num_perm, top_k=top_k)

    def save_clusters_to_json(clusters, similarities, output_file='minhash_clusters_with_similarity.json'):
        """
        Save the clusters and similarities to a JSON file.
        Args:
            clusters (dict): Dictionary of clusters with cluster labels as keys and lists of passwords as values.
            similarities (dict): Dictionary of cluster-to-cluster similarity values.
            output_file (str): Path to the output JSON file.
        """
        # Convert the clusters and similarities to a serializable format
        clusters_serializable = {key: value for key, value in clusters.items()}

        output_data = {
            'Clusters': clusters_serializable,
            'Cluster Similarities': similarities
        }

        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, ensure_ascii=False, indent=4)
        print(f"Clusters and similarities saved to {output_file}")

    # Save clusters and gradual similarities to a JSON file
    save_clusters_to_json(clusters, cluster_similarity, output_file='minhash_clusters_with_similarity.json')

    return clusters, cluster_similarity


if __name__ == "__main__":

    dataset_name = 'rockyou2024-100K.txt'
    passwords, statistics = ps.analyze_passwords(dataset_name)

    """ Cluster strings by entropy and Find and show closest cluster"""
    cluster_strings_by_entropy = cluster_strings_by_entropy(passwords, entropy_threshold=0.5)
    # closest_passwords = find_and_show_closest_cluster("ExamplePassword123", cluster_strings_by_entropy, top_n=10)

    """ Cluster strings with MinHash """
    clusters_minhash = cluster_strings_with_minhash_lsh(passwords, threshold=0.5, num_perm=128)