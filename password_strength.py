import string
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
import pandas as pd
from password_strength_helper import (automatic_deny, almost_deny, give_character_type_score, calc_no_special_password_score,
                                      calc_modified_password_score, remove_pattern)

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'

def calculate_password_strength(password, with_details=False):

    password = password.replace("\n", "")
    if automatic_deny(password):
        if with_details:
            return 0, password, 0, 0, 0
        return 0, password
    if re.fullmatch(email_pattern, password):
        if with_details:
            return 0, password, 0, 0, 0
        return 0, password


    password_score = 3 * give_character_type_score(password)
    unmodified_score = calc_modified_password_score(password)
    p = remove_pattern(password)

    only_letters = re.sub(r'\d{2,}', '', password)
    only_letters = re.sub(r'[^A-Za-z0-9]+', '', only_letters)
    only_letters_score = calc_no_special_password_score(only_letters)

    remove_pattern_score = calc_modified_password_score(p)

    if automatic_deny(only_letters) or almost_deny(password):
        password_score += unmodified_score * 0.25
        final_score = min(max(0, (password_score * 0.3)), 10)

    elif automatic_deny(p):
        password_score += unmodified_score * 0.25
        password_score += only_letters_score * 0.25
        final_score = min(max(0, (password_score * 0.3)), 10)

    else:
        password_score += unmodified_score * 0.25
        password_score += only_letters_score * 0.25
        password_score += remove_pattern_score
        final_score = min(max(0, (password_score * 0.28) - 3.4), 10)

    if with_details:
        return final_score, p, unmodified_score, only_letters_score, remove_pattern_score

    return final_score, p


# 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

#
# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from PIL import Image
# # from website_static_clusters import static_clusters_page
# from website_dynamic_clusters import dynamic_clusters_page
# from website_static_visualizations import static_visualization_page
# from website_dynamic_visualizations import dynamic_visualization_page
# from website_static_strength import static_strength_page
# from website_dynamic_strength import dynamic_strength_page
#
#
# def main():
#     st.set_page_config(layout="wide")  # Set the layout to wide to make the margin thinner
#     st.title('Password Project')
#     page = 'Password Strength Analysis'
#     # Sidebar with page options
#     page = st.sidebar.radio(
#         'Choose a Page', ['Password Analysis: A Visual Dashboard Overview', 'Insert Your Password',
#                           'Clustering Analysis', 'Insert Password for Clustering Analysis',
#                           'Password Strength Statistics', 'Password Strength Analysis'])
#
#     if page == 'Password Analysis: A Visual Dashboard Overview':
#         st.header('Password Analysis: A Visual Dashboard Overview')
#         st.write("This page is intended for statistical analysis.")
#
#         st.subheader('Detailed Analytical Approaches')
#         st.write("""
#         On this page, we conduct a thorough examination of password security using data derived from various breaches. Our analytical framework includes:
#         - **Histogram Analysis**: We create detailed visualizations of password frequency and robustness to pinpoint prevalent vulnerabilities.
#         - **K-Nearest Neighbors (KNN)**: Utilizing advanced KNN techniques, we classify password strengths and model predictions on potential breach impacts.
#         - **Comparative Metrics**: By comparing password security across different data breaches, we derive insights that inform robust cybersecurity strategies.
#
#         Our objective is to offer a nuanced understanding of password security dynamics and to suggest enhanced strategies for safeguarding data against breaches.
#         """)
#
#         static_visualization_page()
#
#     elif page == 'Insert Your Password':
#         st.header('Insert Your Password')
#
#         dynamic_visualization_page()
#         # You can process the password input as needed
#
#     elif page == 'Password Strength Statistics':
#         st.header('Password Strength Statistics')
#
#         static_strength_page()
#
#     elif page == 'Password Strength Analysis':
#         st.header('Password Strength Analysis')
#
#         dynamic_strength_page()
#
#     # elif page == 'Clustering Analysis':
#     #     static_clusters_page()
#
#     # elif page == 'Insert Password for Clustering Analysis':
#     #     dynamic_clusters_page()
#
#
# if __name__ == "__main__":
#     main()






