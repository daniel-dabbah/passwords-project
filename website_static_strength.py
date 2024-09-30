import streamlit as st
import json
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import os
from password_strength import calculate_password_strength
from website_dynamic_strength import dynamic_strength_page

def plot_password_strength_bins():

    dynamic_strength_page()
    st.write("""
        The scatter plot below provides a detailed and engaging look at password strength, with scores ranging from 0 to 10.
        Each dot represents a password, and when you hover over it, you'll see both the original and a modified version, 
        where common words or patterns have been removed. This helps highlight which patterns make passwords more predictable and vulnerable.

        Notice the density of passwords between scores of 1 and 6, indicating many have notable weaknesses. 
        On the other hand, some passwords score a perfect 10, showcasing strong security with no major vulnerabilities, 
        while others, such as email-like passwords, score 0, meaning they can be cracked within minutes using basic brute-force techniques.
    """)

    # with open('password_strength_bins.json', 'r') as f:
    #     bins_list = json.load(f)

    # bins = np.array(bins_list)

    # fig = plt.figure(figsize = (13, 4))

    # plt.bar(range(10), bins, color ='midnightblue')

    # plt.xlabel("Password Strength Score", fontsize=18)
    # plt.ylabel("Number of Passwords", fontsize=18)
    # plt.title("Cracked Passwords Strength Histogram", fontsize=23)
    # plt.xticks(ticks = range(0, 10), labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7",
    #                                          "7-8", "8-9", "9-10"], fontsize=14)
    # st.pyplot(fig)

def plot_password_strength_scatter():
    st.write("""
                 The scatter plot below presents the password strength distribution, offering deeper insights into which passwords score well and which perform poorly. For each password, we also show a modified version (with common words and patterns removed), allowing us to see the specific patterns we aim to avoid. Some passwords score a perfect 10, indicating no notable weaknesses, while others score 0, as they can be cracked in minutes using a simple brute-force algorithm.
            """)
    st.write("""
                     Notice that password that have the pattern of an email address get a 0, 
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

    # json_files_path = ''
    # entropy_json_name = 'entropy_clusters.json'
    #
    # entropy_json_path = os.path.join(json_files_path, entropy_json_name)
    #
    # with open(entropy_json_path, 'r', encoding='utf-8') as json_file:
    #     entropy_clusters = json.load(json_file)
    #
    #
    # cluster_data = []
    # num_clusters = 0
    # for entropy, passwords in entropy_clusters.items():
    #     if len(passwords) > 500 and num_clusters < 20:
    #         num_clusters += 1
    #         sample_passwords = passwords[:3]  # Get up to 3 sample passwords
    #         average_password_strength = 0
    #         for password in passwords[:500]:
    #             average_password_strength += calculate_password_strength(password)[0]
    #         average_password_strength /= 500
    #         cluster_data.append({
    #             'Entropy': float(entropy),
    #             'AVG Strength': round(average_password_strength, 2),
    #             'Cluster Size': len(passwords),
    #             'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
    #             'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
    #             'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
    #         })
    #     if num_clusters >= 20:
    #         break
    #
    # num_clusters = len(cluster_data)
    # df = pd.DataFrame(cluster_data)
    #
    # df = df.sort_values(by='AVG Strength', ascending=False)
    #
    # df.to_csv('password_strength_text_files/entropy_cluster_strength.csv', index=False)

    df = pd.read_csv('password_strength_text_files/entropy_cluster_strength.csv')

    num_clusters = len(df)
    st.dataframe(df[['Entropy', 'AVG Strength', 'Cluster Size', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])

    st.write("""
            The figure below illustrates the strong correlation between the strength and entropy of passwords. \n
            This relationship makes sense, as higher entropy usually indicates greater uncertainty and 
            unpredictability, traits that are essential for strong passwords.
        """)
    correlation = df['Entropy'].corr(df['AVG Strength'])

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize = (12, 5))

    # Plot Entropy with the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Cluster Index')
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

    # json_files_path = ''
    # ngram_json_name = 'ngram_clusters.json'
    # ngram_json_path = os.path.join(json_files_path, ngram_json_name)
    #
    # with open(ngram_json_path, 'r', encoding='utf-8') as json_file:
    #     ngram_clusters = json.load(json_file)
    #
    # # Prepare data for visualization
    # cluster_data = []
    # num_clusters = 0
    # ngram_data_list = []
    # for log_likelihood, cluster in ngram_clusters['Clusters'].items():
    #     average_likelihood = cluster['Average Log Likelihood']
    #     classification = cluster['Classification']
    #     passwords = cluster['Passwords']
    #
    #     # Filter: Only include clusters with log-likelihood > -500 and size > 100
    #     if num_clusters<20 and len(passwords) > 50:
    #
    #         # Extract up to 3 sample passwords
    #         num_clusters += 1
    #         num_samples = min(500, len(passwords))
    #         average_password_strength = 0
    #         for password in passwords[:num_samples]:
    #             average_password_strength += calculate_password_strength(password['Password'])[0]
    #         average_password_strength /= num_samples
    #
    #         sample_passwords = [p['Password'] for p in passwords[:3]]
    #         ngram_data_list.append({
    #             'Log Likelihood': float(log_likelihood),
    #             'Average Log Likelihood': average_likelihood,
    #             'AVG Strength': round(average_password_strength, 2),
    #             'Classification': classification,
    #             'Cluster Size': len(passwords),
    #             'Sample Password 1': sample_passwords[0] if len(sample_passwords) > 0 else '',
    #             'Sample Password 2': sample_passwords[1] if len(sample_passwords) > 1 else '',
    #             'Sample Password 3': sample_passwords[2] if len(sample_passwords) > 2 else ''
    #         })
    #     if num_clusters >= 20:
    #         break
    #
    # df = pd.DataFrame(ngram_data_list)
    #
    # df = df.sort_values(by='AVG Strength', ascending=False)
    #
    # df.to_csv('password_strength_text_files/likelihood_cluster_strength.csv', index=False)

    df = pd.read_csv('password_strength_text_files/likelihood_cluster_strength.csv')
    num_clusters = len(df)

    st.dataframe(df[['Average Log Likelihood', 'AVG Strength', 'Cluster Size', 'Classification', 'Sample Password 1', 'Sample Password 2', 'Sample Password 3']])

    st.write("""
    The figure below illustrates a strong negative correlation between a password's likelihood and its strength. 
    This means that passwords with lower likelihood values, indicating less common or predictable sequences, 
    tend to have higher strength, making them harder to crack with brute-force methods.
    """)
    st.write("""
        * Note that the likelihood scale is inverted in the plot, where higher likelihood values appear lower and lower values are displayed higher. 
        This inversion provides a clearer perspective on the relationship between likelihood and password strength.
    """)

    correlation = df['Average Log Likelihood'].corr(df['AVG Strength'])
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize = (12, 5))

    # Plot Entropy with the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Cluster Index')
    ax1.set_ylabel('Average Log Likelihood', color=color)
    ax1.plot(range(1, num_clusters+1), df['Average Log Likelihood'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    # Invert the scale of the second y-axis
    ax1.set_ylim(ax1.get_ylim()[::-1])  # Reverse the y-axis limits
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

    json_files_path = 'json_files'
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

def plot_cracked_passwords_strength_histogram():
    directory_path = "password_strength_text_files/"

    # file_name = "cracked_passwords.txt"
    # with open(directory_path + file_name, 'r') as f:
    #     cracked_passwords = f.readlines()
    #
    # legit_passwords = set()
    # for p in cracked_passwords:
    #     p = p.replace("\n", "")
    #
    #     # remove too long passwords
    #     if len(p) > 35:
    #         continue
    #
    #     legit_passwords.add(p)
    #
    # cracked_passwords = list(legit_passwords)
    #
    # bins = np.zeros(10)
    # scores = list()
    # for p in cracked_passwords:
    #     score, p1 = calculate_password_strength(p)
    #     scores.append({'Password' : p, 'Score' : score})
    #     if score >= 10:
    #         score = 9.9
    #     bins[int(score)] += 1
    #
    #
    # df = pd.DataFrame(scores)
    #
    # df = df.sort_values(by='Score', ascending=False)
    #
    #
    # df.to_csv('password_strength_text_files/cracked_password_scores.csv', index=False)

    df = pd.read_csv('password_strength_text_files/cracked_password_scores.csv')

    st.dataframe(df[['Password', 'Score']])


    # # Convert NumPy array to a list
    # bins_list = bins.tolist()

    # Save as JSON
    # with open(directory_path + 'cracked_password_strength_bins.json', 'w') as f:
    #     json.dump(bins_list, f)

    with open(directory_path + 'cracked_password_strength_bins.json', 'r') as f:
        bins_list = json.load(f)

    bins = np.array(bins_list)

    fig = plt.figure(figsize = (13, 4))

    plt.bar(range(10), bins, color ='midnightblue')

    plt.xlabel("Password Strength Score", fontsize=15)
    plt.ylabel("Number of Passwords", fontsize=15)
    plt.title("Cracked Passwords Strength Histogram", fontsize=17)
    plt.xticks(ticks = range(0, 10), labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7",
                                             "7-8", "8-9", "9-10"], fontsize=14)

    # st.pyplot(fig)


def static_strength_page():
    st.write("""
                 Try our Password Strength Algorithm! \n
                Our algorithm examines basic features such as the password’s length, number of unique characters, and use of different character types, as well as advanced techniques, such as regular expressions to detect common patterns and predefined lists to identify passwords containing popular words and numbers.
            """)

    plot_password_strength_bins()

    st.subheader('Visualizing the strength of each password')

    plot_password_strength_scatter()

    # st.subheader('Cracked Passwords Strength Examples')

    # plot_cracked_passwords_strength_histogram()

    st.subheader('Measuring Average Password Strengts in Clusters')
    st.write("""
        This section provides insights into the strength of passwords across different clusters, categorized by entropy and n-gram log-likelihood.

        To evaluate our algorithm, we selected 20 random entropy-based clusters and calculated the average entropy and password strength for each cluster.
        As expected, we found a strong positive correlation between these metrics, confirming that higher entropy generally leads to more unpredictable and stronger passwords.

        We also analyzed 16 random clusters based on n-gram log-likelihood, where we observed a strong negative correlation. 
        This shows that passwords with higher log-likelihood scores tend to be weaker and more predictable, 
        as they follow common patterns or linguistic structures that are easier to guess.
    """)

    plot_strength_of_entropy_clusters()

    plot_strength_of_ngram_clusters()

    # plot_strength_of_minhash_clusters()   # commented because not informative


