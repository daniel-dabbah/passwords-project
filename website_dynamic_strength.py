import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
from password_strength import calculate_password_strength


def dynamic_strength_page():
    with open('json_files/password_strength_bins.json', 'r') as f:
        bins_list = json.load(f)

    bins = np.array(bins_list)

    password = st.text_input("Enter a password for analysis",
                             help="Type your password here to see how strong it is.")
    st.caption("Your password will be analyzed locally and not stored or transmitted.")

    scores = calculate_password_strength(password, with_details=True)
    password_score, removed_pattern, unmodified_score, remove_special_score, remove_pattern_score = scores
    rounded_password_score = int(password_score)
    if rounded_password_score >=10:
        rounded_password_score = 9
    fig, ax = plt.subplots(figsize=(10, 2))
    bars = ax.bar(range(10), bins, color='midnightblue')

    ax.set_xticks(ticks=range(0, 10), labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7",
                                           "7-8", "8-9", "9-10"], fontsize=14)
    if password:
        bars[rounded_password_score].set_color('red')
        ax.get_xticklabels()[rounded_password_score].set_color('red')
    
    plt.xlabel("Password Strength Score", fontsize=10)
    plt.ylabel("Number of Passwords", fontsize=10)
    plt.title("Password Scores Histogram", fontsize=15)

    st.pyplot(fig)

    score_with_words = "Very Weak"
    if password_score > 8:
        score_with_words = "Very Strong"
    elif password_score > 6:
        score_with_words = "Strong"
    elif password_score > 4:
        score_with_words = "Medium"
    elif password_score > 2:
        score_with_words = "Weak"
    else:
        score_with_words = "Very Weak"


    if password:
        st.subheader("**Score Breakdown**")
        st.write(f"""
                     **Your Password**: {password}  , your password is {score_with_words} \n 
                     **Your score**: {password_score} \n
                     **Your unmodified score**: {unmodified_score / 3.1} \n
                     **Score of password with removed punctuation and digit sequences**: {remove_special_score / 3} \n
                     **Score of password with common sequences removed**: {remove_pattern_score / 3.1}\n
                     **password with common sequences removed**: {removed_pattern}
                """)





