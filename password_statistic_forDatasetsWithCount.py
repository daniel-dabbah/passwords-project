import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud

# Function to read the password data from a text file
def read_password_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            password, count = line.strip().split('|')
            data.append((password, int(count)))
    return pd.DataFrame(data, columns=['password', 'count'])

# Replace 'your_password_file.txt' with the actual path to your file
password_file_path = 'Pwdb-statistical-lists-100k-occurances.txt'
password_data = read_password_data(password_file_path)

# Plot 1: Histogram of Password Frequencies
plt.figure(figsize=(10, 6))
plt.hist(password_data['count'], bins=50, log=True)
plt.title('Histogram of Password Frequencies')
plt.xlabel('Password Count')
plt.ylabel('Number of Passwords (log scale)')
plt.show()

# Plot 2: Top 20 Most Common Passwords
top_20_passwords = password_data.nlargest(20, 'count')

plt.figure(figsize=(12, 6))
plt.bar(top_20_passwords['password'], top_20_passwords['count'])
plt.title('Top 20 Most Common Passwords')
plt.xlabel('Password')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Plot 3: Password Length Distribution
password_data['length'] = password_data['password'].apply(len)

plt.figure(figsize=(10, 6))
plt.hist(password_data['length'], bins=range(1, 21), align='left')
plt.title('Password Length Distribution')
plt.xlabel('Password Length')
plt.ylabel('Frequency')
plt.show()

