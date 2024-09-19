import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def main():
    st.set_page_config(layout="wide")  # Set the layout to wide to make the margin thinner
    st.title('Password Project')

    # Sidebar with page options
    page = st.sidebar.radio(
        'Choose a Page', ['Password Analysis: A Visual Dashboard Overview', 'Insert Your Password'])

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
        

        # Define the path to the generated plots
        plot_path = 'generated_plots/rockyou_mini.txt'

        # List all plot files in the directory
        plot_files = [f for f in os.listdir(plot_path) if os.path.isfile(os.path.join(plot_path, f))]

        # Display graphs in rows, each containing 3 graphs
        for i in range(0, len(plot_files), 3):  # loop through the plot files in steps of 3
            cols = st.columns(3)  # create 3 columns for each row
            for j in range(3):  # loop to fill each column with a graph
                if i + j < len(plot_files):  # check if there are remaining plot files
                    with cols[j]:
                        plot_file = plot_files[i + j]
                        image = Image.open(os.path.join(plot_path, plot_file))
                        st.image(image, use_column_width=True, width=2400)
        

    elif page == 'Insert Your Password':
        st.header('Insert Your Password')
        st.write("Please enter your password below.")
        st.write("For example, write 'MosheInLondon123'")

        password = st.text_input("Password", type="password")
        # You can process the password input as needed


if __name__ == "__main__":
    main()
