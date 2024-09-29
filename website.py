

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from website_static_clusters import static_clusters_page
from website_static_visualizations import static_visualization_page
from website_dynamic_visualizations import dynamic_visualization_page
from website_static_strength import static_strength_page
from website_dynamic_strength import dynamic_strength_page
from website_introduction import website_introduction


def main():
    st.set_page_config(layout="wide")  
    st.title('How Secure is Your Password?')
    page = 'Password Strength Analysis'
    # Sidebar with page options
    page = st.sidebar.radio(
        'Choose a Page', ['Introduction', 'Breached Password Analysis', 
                          'Check Your Password Statistics',
                          'Check Your Password Strength'])
    
    if page == 'Introduction':
        website_introduction()

    if page == 'Breached Password Analysis':
        st.header('Breached Password Analysis')
        static_visualization_page()

    elif page == 'Check Your Password Statistics':
        st.header('Check Your Password Statistics')
        dynamic_visualization_page()
       
    elif page == 'Check Your Password Strength':
        st.header('Check Your Password Strength')
        static_strength_page()


if __name__ == "__main__":
    main()

