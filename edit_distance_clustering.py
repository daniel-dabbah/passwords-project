# import edit_distance
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from password_statistics import is_only_ascii
from tqdm import tqdm
from difflib import SequenceMatcher
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.figure_factory as ff

def transform_string(s):
    """Transform string such that sequences of lowercase letters become 'a',
       sequences of uppercase letters become 'B', and sequences of digits become '0'."""
    if not s:
        return ""
    transformed = []
    i = 0
    if s[0].islower():
        transformed.append('a')
    elif s[0].isupper():
        transformed.append('B')
    elif s[0].isdigit():
        transformed.append('2')
    else:
        transformed.append('~')

    while i < len(s):
        char = s[i]
        # Check for sequences of lowercase letters
        if char.islower():
            j = 1
            while i < len(s) and s[i].islower():
                i += 1
                j += 1
            transformed.append('l')
            while j > 0:
                transformed.append('l')
                j -= 4
        # Check for sequences of uppercase letters
        elif char.isupper():
            j = 1
            while i < len(s) and s[i].isupper():
                i += 1
                j += 1
            transformed.append('U')
            while j > 0:
                transformed.append('U')
                j -= 4
        # Check for sequences of digits
        elif char.isdigit():
            j = 1
            while i < len(s) and s[i].isdigit():
                i += 1
                j += 1
            transformed.append('1')
            while j > 0:
                transformed.append('1')
                j -= 4
        # For any other character, just add it as is
        else:
            transformed.append('~')
            transformed.append(char)

    if s[-1].islower():
        transformed.append('a')
    elif s[-1].isupper():
        transformed.append('B')
    elif s[-1].isdigit():
        transformed.append('2')
    else:
        transformed.append('~')

    return ''.join(transformed)


# Function to calculate the edit distance
def edit_distance(str1, str2, measure_type_distance = False):

    if measure_type_distance:
        str1 = transform_string(str1)
        str2 = transform_string(str2)

    m, n = len(str1), len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_edit_distance_matrix(strings, measure_type_distance=False):
    n = len(strings)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = edit_distance_for_dendogram(strings[i], strings[j])

            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix


def transform_string_for_dendogram(s):
    """Transform string such that sequences of lowercase letters become 'a',
       sequences of uppercase letters become 'B', and sequences of digits become '0'."""
    if not s:
        return ""
    transformed = [s[0]]
    i = 0
    while i < len(s):
        char = s[i]
        if char.islower():
            transformed.append('l')
            transformed.append(char)
        elif char.isupper():
            transformed.append('U')
            transformed.append(char)
        elif char.isdigit():
            transformed.append('0')
            transformed.append(char)
        else:
            transformed.append('~')
            transformed.append(char)
        i += 1

    return ''.join(transformed)


def edit_distance_for_dendogram(s1, s2):
    s1 = transform_string_for_dendogram(s1)
    s2 = transform_string_for_dendogram(s2)
    m, n = len(s1), len(s2)

    # Create a DP table to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the base case where one string is empty
    for i in range(m + 1):
        dp[i][0] = i  # Cost of deleting all characters in s1 to match an empty s2
    for j in range(n + 1):
        dp[0][j] = j  # Cost of inserting all characters in s2 to match an empty s1

    # Fill in the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost if characters match
            else:
                dp[i][j] = min(dp[i - 1][j],  # Deletion from s1
                               dp[i][j - 1]) + 1  # Insertion into s1

    return dp[m][n]

def compute_distance_matrix_for_dendogram(strings):
    n = len(strings)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = edit_distance_for_dendogram(strings[i], strings[j])
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix


def cluster_strings_by_edit_distance(strings, n_clusters=2, measure_type_distance = False):

    distance_matrix = compute_edit_distance_matrix(strings, measure_type_distance)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')

    labels = clustering.fit_predict(distance_matrix)
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(strings[i])
    return clusters


def distance_evaluator(s1, s2):
    """Compute the edit distance between two strings."""
    matcher = SequenceMatcher(None, s1, s2)
    return 1 - matcher.ratio()  # A ratio closer to 0 means more similar


def best_cluster_for_string(input_string, clusters):
    """
    Given a string and a list of clusters (each cluster is a list of strings),
    return the cluster that best suits the input string.
    """
    best_cluster = None
    best_avg_distance = float('inf')
    for cluster_id, cluster_strings in clusters.items():
        total_distance = 0
        for string in cluster_strings:
            total_distance += edit_distance_for_dendogram(input_string, string)

        avg_distance = total_distance / len(cluster_strings) if cluster_strings else float('inf')

        if avg_distance < best_avg_distance:
            best_avg_distance = avg_distance
            best_cluster = cluster_strings
    return best_cluster

def get_passwords(file_path):
    """Function to analyze the passwords in the file and return the statistics
       We read the file line by line and analyze each password to get the statistics
       and save them in a dictionary. We also store the passwords in a list for further analysis.
    """
    passwords = []
    with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
        for line in tqdm(file, desc="Analyzing passwords"):
            password = line.strip()     # Check password is valid ASCII
            if not is_only_ascii(password): continue
            passwords.append(password)
    return passwords


def create_dendogram_for_cluster(password_list):
    X = compute_distance_matrix_for_dendogram(password_list)
    Z = linkage(X, 'ward')
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=password_list)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

def create_wordcloud_for_cluster(password_list):
    wordcloud = WordCloud(width=800, height=400, include_numbers=True).generate(' '.join(password_list))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Example usage
passwords = get_passwords('DataSets/rockyou2009_sample_10K.txt')
clusters = cluster_strings_by_edit_distance(passwords[100:200], n_clusters=6, measure_type_distance=True)


for i in range(len(clusters)):
    print(f"Cluster {i}: {clusters[i]}")

# example for visualizing a cluster as a wordcloud
create_wordcloud_for_cluster(clusters[1])

# example for creating a dendogram for all the password in a cluster
create_dendogram_for_cluster(clusters[1])

# Example for finding the cluster that suits a string the most
for i in range(0,10):
    print(passwords[i])
    print(best_cluster_for_string(passwords[i], clusters)[0:6])

