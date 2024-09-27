
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
# from website_static_clusters import static_clusters_page
from website_dynamic_clusters import dynamic_clusters_page
from website_static_visualizations import static_visualization_page
from website_dynamic_visualizations import dynamic_visualization_page
from website_static_strength import static_strength_page
from website_dynamic_strength import dynamic_strength_page



def main():
    st.set_page_config(layout="wide")  # Set the layout to wide to make the margin thinner
    st.title('Password Project')
    page = 'Password Strength Analysis'
    # Sidebar with page options
    page = st.sidebar.radio(
        'Choose a Page', ['Password Analysis: A Visual Dashboard Overview', 'Insert Your Password',
                          'Clustering Analysis','Insert Password for Clustering Analysis',
                          'Password Strength Statistics', 'Password Strength Analysis'])

    if page == 'Password Analysis: A Visual Dashboard Overview':
        st.header('Password Analysis: A Visual Dashboard Overview')
        st.write("This page is intended for statistical analysis.")

        st.subheader('Detailed Analytical Approaches')
        st.write("""
        On this page, we conduct a thorough examination of password security using data derived from various breaches. Our analytical framework includes:
        - **Histogram Analysis**: We create detailed visualizations of password frequency and robustness to pinpoint prevalent vulnerabilities.
        - **K-Nearest Neighbors (KNN)**: Utilizing advanced KNN techniques, we classify password strengths and model predictions on potential breach impacts.
        - **Comparative Metrics**: By comparing password security across different data breaches, we derive insights that inform robust cybersecurity strategies.
        
        Our objective is to offer a nuanced understanding of password security dynamics and to suggest enhanced strategies for safeguarding data against breaches.
        """)

        static_visualization_page()

    elif page == 'Insert Your Password':
        st.header('Insert Your Password')
        
        dynamic_visualization_page()
        # You can process the password input as needed

    elif page == 'Password Strength Statistics':
        st.header('Password Strength Statistics')

        static_strength_page()

    elif page == 'Password Strength Analysis':
        st.header('Password Strength Analysis')

        dynamic_strength_page()

    # elif page == 'Clustering Analysis':
    #     static_clusters_page()

    # elif page == 'Insert Password for Clustering Analysis':
    #     dynamic_clusters_page()

if __name__ == "__main__":
    main()
