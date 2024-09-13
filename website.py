import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def main():
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

        # Display 8 line graphs in 4 rows, each containing 2 graphs
        for i in range(4):  # loop through 4 rows
            cols = st.columns(2)  # create 2 columns for each row
            for j in range(2):  # loop to fill each column with a graph
                with cols[j]:
                    fig, ax = plt.subplots()
                    x = np.linspace(0, 10, 100)
                    y = x  # Line equation y = x
                    ax.plot(x, y)
                    ax.set_title(f'Graph {2*i + j + 1}: y = x')
                    st.pyplot(fig)

    elif page == 'Insert Your Password':
        st.header('Insert Your Password')
        st.write("Please enter your password below.")
        st.write("For example, write 'MosheInLondon123'")

        password = st.text_input("Password", type="password")
        # You can process the password input as needed


if __name__ == "__main__":
    main()
