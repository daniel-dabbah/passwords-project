import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re


def website_introduction():

    st.title('Introduction')
    
    st.write('''
    This web application is designed to analyze the strength of passwords and provide insights into the characteristics of different passwords.
    The application uses a dataset of passwords to generate visualizations and statistics on password strength.
    ''')

    st.write('''
    ## Password Dataset
    The dataset used in this application contains a list of passwords obtained from various sources.
    The dataset includes passwords that have been leaked or exposed in data breaches.
    The passwords are stored in plain text format, and the dataset is used for analysis purposes only.
    ''')

    st.write('''
    ## Password Strength Analysis
    The application provides two main types of analysis:
    1. **General Password Analysis**: This analysis includes visualizations and statistics on the dataset of passwords.
    2. **Interactive Password Analysis**: This analysis allows users to input their own passwords for analysis.
    ''')

    st.write('''       
    ## Clustering Analysis
    The application also includes a clustering analysis of passwords based on their similarity.
    The clustering analysis groups passwords into clusters based on their common characteristics.
    ''')

    st.write('''
    ## Password Strength Statistics
    The application provides statistics on the strength of passwords in the dataset.
    The statistics include information on password length, character types, and common patterns.
    ''')

    st.write('''
    ## Password Strength Analysis
    The application analyzes the strength of passwords based on common password strength criteria.
    The analysis includes checks for password length, character types, and common patterns.
    ''')

    st.write('''
    ## Conclusion
    This web application provides a comprehensive analysis of password strength and characteristics.
    Users can explore the visualizations, statistics, and analysis results to gain insights into password security.
    ''')

    st.write('''    

    ## References
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Plotly Documentation](https://plotly.com/python/)
    - [Pandas Documentation](https://pandas.pydata.org/docs/)
    - [Numpy Documentation](https://numpy.org/doc/)
    - [Python Regular Expressions Documentation](https://docs.python.org/3/library/re.html)
    ''')
