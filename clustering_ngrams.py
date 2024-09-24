import json
import math
import random
from collections import defaultdict
from password_statistics import calculate_entropy

# Function to calculate n-gram probabilities from a text file
def calculate_ngram_probabilities(file_path, charset, n=2, smoothing=1e-10):
    ngram_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)

    # Read the file and count n-grams
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().lower()
            for i in range(len(line) - n + 1):
                if all(c in charset for c in line[i:i+n]):
                    ngram = line[i:i+n]
                    ngram_counts[ngram[:-1]][ngram[-1]] += 1
                    total_counts[ngram[:-1]] += 1

    # Normalize probabilities with smoothing
    ngram_probs = defaultdict(dict)
    for prev_ngram in ngram_counts:
        for char in charset:
            ngram_probs[prev_ngram][char] = (ngram_counts[prev_ngram][char] + smoothing) / (total_counts[prev_ngram] + smoothing * len(charset))

    return ngram_probs

# Function to calculate log-likelihood for a given string using n-grams
def log_likelihood_string(s, ngram_probs, charset, n=2):
    log_likelihood_value = 0.0
    for i in range(len(s) - n + 1):
        ngram = s[i:i+n]
        if all(c in charset for c in ngram):
            prev_ngram = ngram[:-1]
            next_char = ngram[-1]
            log_likelihood_value += math.log(ngram_probs.get(prev_ngram, {}).get(next_char, 1e-10))  # Use smoothing for unseen n-grams
    return log_likelihood_value

# Function to calculate threshold
def calculate_threshold(good_probs, bad_probs):
    return (min(good_probs) + max(bad_probs)) / 2

# Function to generate random gibberish passwords
def generate_gibberish_dataset(num_passwords=100000, min_len=8, max_len=15, charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890~!@#$%^&*()_+{}[]:\";'<>?,./\\|` "):
    gibberish_passwords = []
    for _ in range(num_passwords):
        password_length = random.randint(min_len, max_len)
        password = ''.join(random.choices(charset, k=password_length))
        gibberish_passwords.append(password)
    return gibberish_passwords

# Function to cluster and classify passwords, and save results to JSON files
def cluster_passwords(meaningful_file_path, clustering_file_path, gibberish_count=100000, output_json='ngram_classification_results.json', cluster_json='ngram_clusters.json'):
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890~!@#$%^&*()_+{}[]:\";'<>?,./\\|` "  # Define the set of allowed characters
    n = 2  # Use 2-grams

    # Calculate n-gram probabilities for the meaningful dataset
    meaningful_probs = calculate_ngram_probabilities(meaningful_file_path, charset, n)

    # Load passwords to be clustered from the specified file
    with open(clustering_file_path, 'r', encoding='utf-8') as file:
        clustering_passwords = [line.strip() for line in file.readlines()]

    # Calculate log-likelihoods
    good_log_likelihoods = [log_likelihood_string(s.lower(), meaningful_probs, charset, n) for s in open(meaningful_file_path, 'r', encoding='utf-8')]
    bad_log_likelihoods = [log_likelihood_string(s.lower(), meaningful_probs, charset, n) for s in clustering_passwords]

    # Calculate the threshold
    threshold = calculate_threshold(good_log_likelihoods, bad_log_likelihoods)

    # Classify passwords and prepare results for JSON
    results = []
    clusters = defaultdict(list)

    for password in clustering_passwords:
        log_likelihood = log_likelihood_string(password, meaningful_probs, charset, n)
        entropy_value = calculate_entropy(password)  # Calculate entropy for each password
        is_meaningful = log_likelihood > threshold
        results.append({
            'Password': password,
            'Log Likelihood': log_likelihood,
            'Entropy': entropy_value,  # Save entropy
            'Classification': 'Meaningful' if is_meaningful else 'Gibberish'
        })
        # Add password to appropriate cluster based on log-likelihood with distance threshold
        found_cluster = False
        for cluster_likelihood in list(clusters.keys()):
            if abs(log_likelihood - cluster_likelihood) <= 5:  # Use a distance threshold of 5
                clusters[cluster_likelihood].append({
                    'Password': password,
                    'Log Likelihood': log_likelihood,
                    'Entropy': entropy_value,  # Save entropy in the cluster too
                    'Classification': 'Meaningful' if is_meaningful else 'Gibberish'
                })
                found_cluster = True
                break
        
        if not found_cluster:
            clusters[log_likelihood].append({
                'Password': password,
                'Log Likelihood': log_likelihood,
                'Entropy': entropy_value,  # Save entropy in the new cluster
                'Classification': 'Meaningful' if is_meaningful else 'Gibberish'
            })

    # Prepare clusters with average log-likelihoods and classification
    cluster_summary = {}
    for log_likelihood, passwords in clusters.items():
        avg_likelihood = sum(p['Log Likelihood'] for p in passwords) / len(passwords)
        avg_entropy = sum(p['Entropy'] for p in passwords) / len(passwords)  # Calculate average entropy for the cluster
        classification = 'Meaningful' if avg_likelihood > threshold else 'Gibberish'
        cluster_summary[log_likelihood] = {
            'Average Log Likelihood': avg_likelihood,
            'Average Entropy': avg_entropy,  # Store average entropy
            'Classification': classification,
            'Passwords': passwords
        }

    # Save classification results to JSON
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    # Save clusters to JSON
    output_data = {
        'Threshold': threshold,
        'Clusters': cluster_summary
    }
    
    with open(cluster_json, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    # Save ngram probabilities to JSON
    with open('ngram_probs.json', 'w', encoding='utf-8') as json_file:
        json.dump(meaningful_probs, json_file, ensure_ascii=False, indent=4)

    print(f"Classification results saved to {output_json}")
    print(f"Clusters saved to {cluster_json}")
    return results, threshold

# Example usage
if __name__ == "__main__":
    meaningful_file_path = "rockyou2009_100K.txt"  # Path to the file containing meaningful passwords
    clustering_file_path = "rockyou2024-100K.txt"  # Path to the file containing passwords to be classified
    clustered_results, threshold = cluster_passwords(meaningful_file_path, clustering_file_path)
    
    print(f"Threshold for classification: {threshold:.4f}")
