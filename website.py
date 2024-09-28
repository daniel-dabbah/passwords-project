

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from website_static_clusters import static_clusters_page
from website_dynamic_clusters import dynamic_clusters_page
from website_static_visualizations import static_visualization_page
from website_dynamic_visualizations import dynamic_visualization_page
from website_static_strength import static_strength_page
from website_dynamic_strength import dynamic_strength_page


def main():
    st.set_page_config(layout="wide")  
    st.title('Password Project')
    page = 'Password Strength Analysis'
    # Sidebar with page options
    page = st.sidebar.radio(
        'Choose a Page', ['Password Analysis: A Visual Dashboard Overview', 'Interactive Password Analysis',
                          'Clustering Analysis', 'Insert Password for Clustering Analysis',
                          'Password Strength Statistics', 'Password Strength Analysis'])

    if page == 'Password Analysis: A Visual Dashboard Overview':
        st.header('Password Analysis: A Visual Dashboard Overview')
        static_visualization_page()

    elif page == 'Interactive Password Analysis':
        st.header('Interactive Password Analysis')
        dynamic_visualization_page()
       

    elif page == 'Password Strength Statistics':
        st.header('Password Strength Statistics')
        static_strength_page()

    elif page == 'Password Strength Analysis':
        st.header('Password Strength Analysis')
        dynamic_strength_page()

    elif page == 'Clustering Analysis':
        st.header('Clustering Analysis')
        static_clusters_page()

    elif page == 'Insert Password for Clustering Analysis':
        st.header('Insert Password for Clustering Analysis')
        dynamic_clusters_page()


if __name__ == "__main__":
    main()

