import streamlit as st
import json
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import os
import mpld3
from password_strength import calculate_password_strength

def plot_password_strength_bins():

    st.write("""
                 The following Histogram visualizes the distribution of password strengths. 
                 We can see that many people use a rather simple password but 
                 most of them do try (or are forced by the security demands to incorporate different character types)
                 to make the password a bit more complex.
            """)
    with open('password_strength_bins.json', 'r') as f:
        bins_list = json.load(f)

    bins = np.array(bins_list)

    fig = plt.figure(figsize = (13, 4))

    plt.bar(range(10), bins, color ='midnightblue')

    plt.xlabel("Password Strength Score", fontsize=18)
    plt.ylabel("Number of Passwords", fontsize=18)
    plt.title("Password Scores Histogram", fontsize=23)
    plt.xticks(ticks = range(0, 10), labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7",
                                             "7-8", "8-9", "9-10"], fontsize=14)


    st.pyplot(fig)

def plot_password_strength_scatter():
    st.write("""
                 The scatter plot below we show the distibution in a different way that allows us to explore 
                 the strength of passwords with more depth and to manually analyze what kind of passwords
                 achieve a good score and which get a bad score. \n
                 For each password we can also see the 
                 modified version of the password (where we removed common words and patterns) which can
                 give an insight to what kind of patters we are looking for.
                 \n
                 We can also see that some passwords get a score of zero as they can be cracked in a 
                 matter of minutes using a simple predefined brute force algorithm. 
                 \n
                 Also we see that a few passwords achieve a perfect 10, this are passwords for which we 
                 cannot point to any significant weaknesses.
            """)

    st.write("""
                     * notice that password that have the pattern of an email address get a 0, 
                     because the password is likely identical or very similar to the users email. 
                """)
    df = pd.read_csv('password_strength_dataframe.csv')

    df['Index'] = df.index

    fig = px.scatter(df, x='Index', y='score', hover_data=['password', 'modified'])

    fig.update_layout(
        width=1300,  # Set width in pixels
        height=700,  # Set height in pixels
        # title="Passwords Strengths",
        # title_font_size = 25,
        font_color="black",
        # title_xanchor='center',
        # title_xref = "paper",
        # title_x=0.5,
        xaxis=dict(tickvals=[],
            title='',
        #     titlefont=dict(color='black', size=21)  # Change x-axis title font color to blue
        ),
        yaxis=dict(
            title='Password Strength',
            titlefont=dict(color='black', size=21),  # Change y-axis title font color to red
            tickfont=dict(color='black', size=16)
        ),
        margin=dict(l=5, r=5, t=5, b=5)  # Reduce margins
    )

    st.plotly_chart(fig)

def plot_strength_of_entropy_clusters():

    st.subheader('Entropy Clusters Table')

    st.write("""Might take a few seconds to load
    """)

    json_files_path = ''
    entropy_json_name = 'entropy_clusters.json'

    entropy_json_path = os.path.join(json_files_path, entropy_json_name)

    with open(entropy_json_path, 'r', encoding='utf-8') as json_file:
        entropy_clusters = json.load(json_file)


    cluster_data = []
    num_clusters = 0
    for entropy, passwords in entropy_clusters.items():
        if len(passwords) > 300 and num_clusters < 20:
            num_clusters += 1
            sample_passwords = passwords[:3]  # Get up to 3 sample passwords
            average_password_strength = 0
            for password in passwords[:300]:
                average_password_strength += calculate_password_strength(password)[0]
            average_password_strength /= 300
            # if average_password_strength < 0.1:
            #     continue
            cluster_data.append({
                'Entropy': float(entropy),
                'AVG Strength': round(average_password_strength, 2),
                'Cluster Size': len(passwords),
                'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
                'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
                'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
            })
        if num_clusters >= 20:
            break

    num_clusters = len(cluster_data)
    df = pd.DataFrame(cluster_data)

    df = df.sort_values(by='AVG Strength', ascending=False)

    st.dataframe(df[['Entropy', 'AVG Strength', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])

    correlation = df['Entropy'].corr(df['AVG Strength'])

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize = (12, 5))

    # Plot Entropy with the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Cluster Strength Rank')
    ax1.set_ylabel('Entropy', color=color)
    ax1.plot(range(1, num_clusters+1), df['Entropy'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(1, num_clusters+1))

    # Create a secondary y-axis to plot AVG Strength
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Strength', color=color)
    ax2.plot(range(1, num_clusters+1), df['AVG Strength'], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title and grid
    plt.title(f'Entropy and Average Strength, Correlation = {correlation:.3f}')
    ax1.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_strength_of_ngram_clusters():

    st.subheader('Ngram Clusters Table')

    json_files_path = ''
    ngram_json_name = 'ngram_clusters.json'
    ngram_json_path = os.path.join(json_files_path, ngram_json_name)

    with open(ngram_json_path, 'r', encoding='utf-8') as json_file:
        ngram_clusters = json.load(json_file)

    # Prepare data for visualization
    cluster_data = []
    num_clusters = 0
    ngram_data_list = []
    for log_likelihood, cluster in ngram_clusters['Clusters'].items():
        average_likelihood = cluster['Average Log Likelihood']
        classification = cluster['Classification']
        passwords = cluster['Passwords']

        # Filter: Only include clusters with log-likelihood > -500 and size > 100
        if num_clusters<20 and len(passwords) > 100:

            # Extract up to 3 sample passwords
            num_clusters += 1
            num_samples = min(300, len(passwords))
            average_password_strength = 0
            for password in passwords[:num_samples]:
                average_password_strength += calculate_password_strength(password['Password'])[0]
            average_password_strength /= num_samples

            sample_passwords = [p['Password'] for p in passwords[:3]]
            ngram_data_list.append({
                'Log Likelihood': float(log_likelihood),
                'Average Log Likelihood': average_likelihood,
                'AVG Strength': round(average_password_strength, 2),
                'Classification': classification,
                'Cluster Size': len(passwords),
                'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
                'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
                'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
            })
        if num_clusters >= 20:
            break


    num_clusters = len(ngram_data_list)
    df = pd.DataFrame(ngram_data_list)

    df = df.sort_values(by='AVG Strength', ascending=False)

    st.dataframe(df[['Average Log Likelihood', 'AVG Strength', 'Cluster Size', 'Classification', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])


    correlation = df['Average Log Likelihood'].corr(df['AVG Strength'])
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize = (12, 5))

    # Plot Entropy with the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Cluster Strength Rank')
    ax1.set_ylabel('Average Log Likelihood', color=color)
    ax1.plot(range(1, num_clusters+1), df['Average Log Likelihood'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.set_xticks(range(1, num_clusters+1))

    # Create a secondary y-axis to plot AVG Strength
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Strength', color=color)
    ax2.plot(range(1, num_clusters+1), df['AVG Strength'], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title and grid
    plt.title(f'Average Log Likelihood and Average Strength, Correlation = {correlation:.3f}')
    ax1.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_strength_of_minhash_clusters():

    st.subheader('Minhash Clusters Table')

    json_files_path = ''
    minhash_json_name = 'minhash_clusters.json'

    minhash_json_path = os.path.join(json_files_path, minhash_json_name)

    with open(minhash_json_path, 'r', encoding='utf-8') as json_file:
        minhash_clusters = json.load(json_file)

    minhash_cluster_data = []
    num_clusters = 0
    for cluster_label, passwords in minhash_clusters.items():
        cluster_size = len(passwords)
        if num_clusters < 10 and cluster_size < 100000 and cluster_size >= 100:
            num_clusters += 1
            average_password_strength = 0
            for password in passwords[:300]:
                average_password_strength += calculate_password_strength(password)[0]
            average_password_strength /= 300

            sample_passwords = passwords[:3]
            minhash_cluster_data.append({
                'Cluster Label': cluster_label,
                'AVG Strength': round(average_password_strength, 2),
                'Cluster Size': cluster_size,
                'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
                'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
                'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
            })
        if num_clusters >= 10:
            break


    df = pd.DataFrame(minhash_cluster_data)

    df = df.sort_values(by='AVG Strength', ascending=False)

    st.dataframe(df[['Cluster Label', 'AVG Strength', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])


def static_strength_page():
    st.write("""
                 In our project we defined a method for evaluating the strength of passwords. 
                 the method is on the conclusions we have derived from the analysis we performed on our password
                 datasets and from the statistics we have gathered. \n
                 he method mainly looks at some simple features of the password such as the length of the password, 
                 the number of different characters, and whether it uses different character types such as 
                 lowercase and uppercase letters, digits, punctuation and other symbols, 
                 where we consider passwords that include more types and more different characters as 
                 better and harder to crack. The method also tries to detect common words and patterns and 
                 penalizes the score for any such word or pattern that we have managed to detect.
            """)

    plot_password_strength_bins()

    st.subheader('Visualizing the strength of each password')

    plot_password_strength_scatter()

    st.subheader('Measuring the average password strength of different clusters')

    plot_strength_of_entropy_clusters()

    plot_strength_of_ngram_clusters()

    # plot_strength_of_minhash_clusters()   # not very informative
