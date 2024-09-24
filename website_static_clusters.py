import streamlit as st
import json
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

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
                     title='Correlation between Average Log-Likelihood and Entropy',
                     labels={'Average Log Likelihood': 'Average Log Likelihood', 'Average Entropy': 'Average Entropy'},
                     size='Cluster Size',  # Use cluster size to scale marker size
                     color_discrete_map={'Meaningful': 'green', 'Gibberish': 'red'})

    # Display the plot in Streamlit
    st.plotly_chart(fig)


def plot_entropy_clusters(entropy_clusters):
    # Prepare data for visualization
    cluster_data = []
    for entropy, passwords in entropy_clusters.items():
        # Prepare sample passwords for the table
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
                     title="Password Clusters by Entropy",
                     labels={'Entropy': 'Entropy Value', 'Cluster Size': 'Number of Passwords'},
                     size='Cluster Size', size_max=20,
                     color='Cluster Size',  # Color by cluster size
                     color_continuous_scale=px.colors.sequential.Viridis)  # Optional color scale

    # Update the hover data to include entropy value, cluster size, and sample passwords
    fig.update_traces(
        hovertemplate=(
            '<b>Entropy:</b> %{x:.2f}<br>'  # Display entropy value
            '<b>Cluster Size:</b> %{y}<br>'  # Display number of passwords
            '<b>Sample Passwords:</b><br>%{hovertext}'  # Display sample passwords
            '<extra></extra>'
        ),
        hovertext=df[['Sample Password 1', 'Sample Password 2', 'Sample Password 3']].apply(lambda x: '<br>'.join(x), axis=1)
    )

    st.plotly_chart(fig)  # Display the Plotly chart

    st.write("""
             The scatter plot above visualizes the clustering of passwords based on their entropy values. \n
                Each point represents a cluster of passwords with a specific entropy value. 
                The size of the point corresponds to the number of passwords in the cluster. 
                Hover over the points to view the entropy value, cluster size, and sample passwords.  \n

                The table below provides a detailed view of the clusters, including the entropy value, cluster size, and sample passwords. 
                The sample passwords are representative of the passwords in each cluster. 
                The clusters in the table are sorted by entropy in ascending order. 
                you can explore the distribution of password entropy values and the corresponding cluster sizes. \n
             
                It can be seen that most passwords fall within the lower entropy range, indicating weaker password strength. \n
        """)

    # Optional: Display the clusters and sample passwords in a table below the plot
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
        sample_passwords = [p['Password'] for p in passwords[:3]]  # Get up to 3 sample passwords
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
    ngram_df = ngram_df.sort_values(by='Log Likelihood', ascending=False)  # Sort DataFrame by Log Likelihood

    # Create an interactive scatter plot for n-gram clusters
    ngram_fig = px.scatter(ngram_df, x='Log Likelihood', y='Cluster Size',
                           title=f"Password Clusters by N-gram Log Likelihood (Threshold: {threshold:.2f})",
                           labels={'Log Likelihood': 'Log Likelihood Value', 'Cluster Size': 'Number of Passwords'},
                           size='Cluster Size', size_max=20,
                           color='Cluster Size', 
                           color_continuous_scale=px.colors.sequential.Viridis)

    ngram_fig.update_traces(
        hovertemplate=(
            '<b>Average Log Likelihood:</b> %{customdata[0]:.2f}<br>'
            '<b>Classification:</b> %{customdata[1]}<br>'
            '<b>Cluster Size:</b> %{y}<br>'
            '<extra></extra>'
        ),
        customdata=ngram_df[['Average Log Likelihood', 'Classification']].values,
    )

    # Add a vertical line for the threshold
    ngram_fig.add_vline(x=threshold, line_color='red', line_dash="dash", annotation_text="Threshold", annotation_position="top right")

    st.plotly_chart(ngram_fig)  # Display the Plotly chart

    # Detailed N-gram Clusters
    st.subheader('N-gram Clusters Table')
    st.dataframe(ngram_df[['Average Log Likelihood', 'Classification', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])


def plot_minhash_clusters(minhash_clusters):
    # Prepare data for MinHash clusters visualization
    minhash_cluster_data = []
    for cluster_label, passwords in minhash_clusters.items():
        # Get up to 3 sample passwords for the table
        sample_passwords = passwords[:3]
        minhash_cluster_data.append({
            'Cluster Label': cluster_label,
            'Cluster Size': len(passwords),
            'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
            'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
            'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
        })

    # Create a DataFrame for MinHash clusters
    minhash_df = pd.DataFrame(minhash_cluster_data)

    # Create a scatter plot for MinHash clusters colored by cluster size
    minhash_fig = px.scatter(minhash_df, x='Cluster Label', y='Cluster Size',
                            title="Password Clusters by MinHash",
                            labels={'Cluster Size': 'Number of Passwords'},
                            size='Cluster Size', size_max=20,
                            color='Cluster Size',  # Color by cluster size
                            color_continuous_scale=px.colors.sequential.Viridis)  # Optional color scale

    # Update the hover data for MinHash clusters
    minhash_fig.update_traces(
        hovertemplate=(
            '<b>Cluster:</b> %{x}<br>'  # Display cluster label
            '<b>Cluster Size:</b> %{y}<br>'  # Display number of passwords
            '<b>Sample Passwords:</b><br>%{hovertext}'  # Display sample passwords
            '<extra></extra>'
        ),
        hovertext=minhash_df[['Sample Password 1', 'Sample Password 2', 'Sample Password 3']].apply(lambda x: '<br>'.join(x), axis=1)  # Unique sample passwords for each cluster
    )

    # Remove the legend and x-axis labels
    minhash_fig.for_each_trace(lambda t: t.update(legendgroup=None))
    minhash_fig.update_layout(xaxis_title="", showlegend=False)

    # Remove x-axis tick labels
    minhash_fig.update_xaxes(tickvals=[], ticktext=[])

    # Display the MinHash plot
    st.plotly_chart(minhash_fig)  # Display the MinHash scatter plot

    # Optional: Display the clusters and sample passwords in a table below the plot
    st.subheader('MinHash Clusters Table')
    st.dataframe(minhash_df[['Cluster Label', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])


def static_clusters_page():

    json_files_path = ''
    entropy_json_name = 'entropy_clusters.json'
    ngram_json_name = 'ngram_clusters.json'
    minhash_json_name = 'minhash_clusters.json'

    entropy_json_path = os.path.join(json_files_path, entropy_json_name)
    ngram_json_path = os.path.join(json_files_path, ngram_json_name)
    minhash_json_path = os.path.join(json_files_path, minhash_json_name)

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
    st.write("Clustering is a type of unsupervised learning that groups data points into clusters based on their similarities.")

    """ Entropy Clustering """
    st.header('Clustering by Entropy')
    st.write("""
             Entropy represents the measure of unpredictability or randomness in a password. \n
             it reflects how resistant a password is to being guessed or cracked. \n
             the more diverse the character set used— including lowercase letters, uppercase letters, numbers, and special symbols, \n
             the higher the entropy, making the password significantly more secure. 
             the entropy is calculated using the formula: entropy = log2(possible combinations). \n

             to ensure robust security, it is recommended that a password achieve at least 80 bits of entropy, \n
             typically requiring a length of at least 12 characters that utilize a blend of all character types. \n
        """)
    plot_entropy_clusters(entropy_clusters)
    

    """ N-Gram Clustering """
    st.header('Clustering by N-gram Log-Likelihood')
    st.write("""
              N-gram Model: We break down passwords into sequences of 2 characters (bi-grams) to capture common patterns in character combinations.\n
                Log-likelihood Calculation: Each password is evaluated against the n-gram model, calculating its likelihood based on how closely it matches common patterns from a meaningful password dataset.\n
             Clustering: Passwords with similar log-likelihoods are grouped together into clusters, where each cluster represents passwords that have similar structural patterns. \n
                Classification: Clusters are classified as either 'Meaningful' or 'Gibberish' based on their average log-likelihood values. \n
                Threshold: A threshold is set to distinguish between 'Meaningful' and 'Gibberish' clusters. \n
                The scatter plot below visualizes the clustering of passwords based on their n-gram log-likelihood values. \n
                Each point represents a cluster of passwords with a specific log-likelihood value. \n
                The size of the point corresponds to the number of passwords in the cluster. \n
                Hover over the points to view the average log-likelihood, classification, cluster size, and sample passwords. \n
                The red dashed line represents the threshold used to classify clusters as 'Meaningful' or 'Gibberish'. \n
                The table below provides a detailed view of the clusters, including the average log-likelihood, classification, cluster size, and sample passwords. \n
                The clusters in the table are sorted by log-likelihood in descending order. \n
                You can explore the distribution of n-gram log-likelihood values and the corresponding cluster sizes. \n
             """)
    plot_ngram_clusters(ngram_clusters)
    

    """ MinHash Clustering """
    st.header('Clustering by MinHash')
    st.write("""
            MinHash is a technique used to efficiently estimate the similarity between sets. 
            For passwords, it allows us to compare them based on the similarity of their character sets.
            Hashing Passwords: Each password is hashed into a compact MinHash signature, capturing the essential information of its character set. \n
            Similarity Calculation: Passwords with similar MinHash signatures are grouped into clusters. 
            These clusters represent groups of passwords with overlapping character sets, making them structurally similar.\n
            The scatter plot below visualizes the clustering of passwords based on their MinHash signatures. \n
            Each point represents a cluster of passwords with a specific label. \n
            The size of the point corresponds to the number of passwords in the cluster. \n
            Hover over the points to view the cluster label, cluster size, and sample passwords. \n
            The table below provides a detailed view of the clusters, including the cluster label, cluster size, and sample passwords. \n
            You can explore the distribution of MinHash clusters and the corresponding cluster sizes. \n
             """)

    plot_minhash_clusters(minhash_clusters)


    """ Entropy vs Log-Likelihood Correlation """
    st.header('Entropy vs Log-Likelihood Correlation')
    st.write("""Correlation between average entropy and average log-likelihood for each cluster.
            The scatter plot below visualizes the relationship between average entropy and average log-likelihood values for each cluster.
            Each point represents a cluster of passwords with a specific entropy and log-likelihood value.
            We can observe that clusters with high entropy values tend to have lower log-likelihood values, 
            indicating a correlation between password strength (entropy) and predictability (log-likelihood).
             """)
    plot_entropy_vs_likelihood_by_cluster(ngram_json_path)