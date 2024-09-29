import string
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
import pandas as pd
import json
from password_strength_helper import (automatic_deny, give_character_type_score,
                                      calc_modified_password_score, remove_pattern)
from password_strength import calculate_password_strength

directory_path = "password_strength_text_files/"
# file_name = "rockyou2009_sample_10K.txt"
file_name = "rockyou2009_20k.txt"
with open(directory_path + file_name, 'r') as f:
    rockyou2009_10k = f.readlines()

legit_passwords = set()
for p in rockyou2009_10k:
    p = p.replace("\n", "")

    # remove too long passwords
    if len(p) > 35 :
        continue

    # remove passwords containing only digits
    if len(p) == sum(1 for char in p if char.isdigit()) and len(p) < 12:
        continue

    legit_passwords.add(p)

rockyou2009_10k = list(legit_passwords)


password_scores = list()
ok_passwords = list()
fixed_passwords = list()
pass_len = list()
char_type_counter = np.zeros(10)
diff_char_counter = np.zeros(40)
length_counter = np.zeros(40)
for p1 in rockyou2009_10k:
    if len(p1) > 32:
        continue
    password_score, p = calculate_password_strength(p1, with_details=False)
    password_scores.append(password_score)
    ok_passwords.append(p1)
    fixed_passwords.append(p)
    pass_len.append(len(p1))

print(len(ok_passwords))
print(f"min score: \t\t{min(password_scores)}")
print(f"max score: \t\t{max(password_scores)}")
print(f"average score:\t{sum(password_scores) / len(password_scores)}")

# Create DataFrame
df = pd.DataFrame()
df['password'] = ok_passwords
df['score'] = password_scores
df['modified'] = fixed_passwords
df['length'] = pass_len

# Save as a CSV file
df.to_csv('password_strength_dataframe.csv', index=False)  # `index=False` avoids saving the index as a column
# fig = px.scatter(df, x='length', y='score', hover_data=['password', 'modified'])
#
# fig.update_layout(
#     title="Passwords Strengths",
#     title_font_size = 25,
#     title_xanchor='center',
#     title_xref = "paper",
#     title_x=0.5,
#     paper_bgcolor="LightSteelBlue",
# )
#
# fig.update_xaxes(title_text="Password Length", title_font_size=20)
# fig.update_yaxes(title_text="Password Strength", title_font_size=20)
# fig.show()


sorted_passwords = sorted(rockyou2009_10k, key=calculate_password_strength, reverse=True)
for i in range(100):
    score , p = calculate_password_strength(sorted_passwords[i])
    print(f"{score:.2f} : {sorted_passwords[i]}")

for i in range(1, 101):
    score , p = calculate_password_strength(sorted_passwords[-i])
    print(f"{score:.2f} : {sorted_passwords[-i]}")

bins = np.zeros(10)
for p in sorted_passwords:
    score, p1 = calculate_password_strength(p)
    if score >= 10:
        score = 9.9
    bins[int(score)] += 1

# Convert NumPy array to a list
bins_list = bins.tolist()

# Save as JSON
with open('password_strength_bins.json', 'w') as f:
    json.dump(bins_list, f)
#
# fig = plt.figure(figsize = (15, 10))
#
# # creating the bar plot
# plt.bar(range(10), bins, color ='maroon')
#
# plt.xlabel("Password Strength Score", fontsize=18)
# plt.ylabel("Number of Passwords", fontsize=18)
# plt.title("Password Scores Histogram", fontsize=23)
# plt.xticks(ticks = range(0, 10), labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"], fontsize=14)

# plt.show()
