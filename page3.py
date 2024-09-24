import streamlit as st

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide to make the margin thinner
    st.title('Password Project')

    st.header('Password Analysis Dashboard')
    
    st.write("Welcome to our Password Analysis Dashboard. This page provides a basic overview for testing purposes.")

    st.subheader('Features:')
    st.markdown("""
    - Histogram Analysis
    - Category Breakdown
    - Position Analysis
    """)

    st.info("More detailed visualizations and analysis will be added in future updates.")
    
main()